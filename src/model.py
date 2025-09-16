from typing import List, Optional, Tuple

import copy
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import SGD, Adam, AdamW
from .utils import SVGD, RBF, FHBI
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import MulticlassCalibrationError as CalibrationError
from torchmetrics.classification.stat_scores import StatScores
from transformers import AutoConfig, AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup
import timm

from src.loss import SoftTargetCrossEntropy
from src.mixup import Mixup
from .utils import block_expansion
from .lora import LoRA_ViT
from .base_vit2 import ViT, CustomLinear, CustomLinear2
# from .base_vit import ViT, CustomLinear

torch.autograd.set_detect_anomaly(True)


MODEL_DICT = {
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",
    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
}


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vit-b16-224-in21k",
        optimizer: str = "sgd",
        rho: float= 0.05,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        n_classes: int = 10,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        mix_prob: float = 1.0,
        label_smoothing: float = 0.0,
        image_size: int = 224,
        weights: Optional[str] = None,
        training_mode: str = "full",
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_target_modules: List[str] = ["query", "value"],
        lora_dropout: float = 0.0,
        lora_bias: str = "none",
        from_scratch: bool = False,
        num_particles: int = 10,
        use_sam: bool = False,
        weights_path: str = 'checkpoint/B_16.pth',
        epsilon: float = 0.01,
        cov_mat: bool = True,
        max_num_models: int = 20,
        sigma = None,
        base_optimizer_name = "sgd", 
    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint. List found in src/model.py
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_steps: Number of warmup steps
            n_classes: Number of target class
            mixup_alpha: Mixup alpha value
            cutmix_alpha: Cutmix alpha value
            mix_prob: Probability of applying mixup or cutmix (applies when mixup_alpha and/or
                cutmix_alpha are >0)
            label_smoothing: Amount of label smoothing
            image_size: Size of input images
            weights: Path of checkpoint to load weights from (e.g when resuming after linear probing)
            training_mode: Fine-tuning mode. One of ["full", "linear", "lora"]
            lora_r: Dimension of LoRA update matrices
            lora_alpha: LoRA scaling factor
            lora_target_modules: Names of the modules to apply LoRA to
            lora_dropout: Dropout probability for LoRA layers
            lora_bias: Whether to train biases during LoRA. One of ['none', 'all' or 'lora_only']
            from_scratch: Initialize network with random weights instead of a pretrained checkpoint
        """
        super().__init__()
        # self.automatic_optimization = False
        self.save_hyperparameters()
        self.model_name = model_name
        self.optimizer = optimizer
        self.rho = rho
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.n_classes = n_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.label_smoothing = label_smoothing
        self.image_size = image_size
        self.weights = weights
        self.training_mode = training_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.from_scratch = from_scratch
        self.num_particles =  num_particles
        self.use_sam = use_sam
        self.epsilon = epsilon
        self.cov_mat = cov_mat
        self.max_num_models = max_num_models
        self.sigma = sigma
        self.base_optimizer_name = base_optimizer_name
        # Initialize network
        try:
            model_path = MODEL_DICT[self.model_name]
        except:
            raise ValueError(
                f"{model_name} is not an available model. Should be one of {[k for k in MODEL_DICT.keys()]}"
            )

        if self.from_scratch:
            # Initialize with random weights
            config = AutoConfig.from_pretrained(model_path)
            config.image_size = self.image_size
            self.net = AutoModelForImageClassification.from_config(config)
            self.net.classifier = torch.nn.Linear(config.hidden_size, self.n_classes)
        else:
            # Initialize with pretrained weights
            if self.optimizer in ['svgd', "deep_ens", "fhbi"]:
                print('Model name', self.model_name)
                self.net = ViT(name='vit-b16-224-in21k', pretrained=True, num_classes=self.n_classes, image_size=self.image_size, num_particles=self.num_particles, weight_path=weights_path)
                
                self.net = self.net.cuda()
                
        # Load checkpoint weights
        if self.weights:
            print(f"Loaded weights from {self.weights}")
            ckpt = torch.load(self.weights)["state_dict"]

            # Remove prefix from key names
            new_state_dict = {}
            for k, v in ckpt.items():
                if k.startswith("net"):
                    k = k.replace("net" + ".", "")
                    new_state_dict[k] = v

            self.net.load_state_dict(new_state_dict, strict=True)
            
        # Prepare model depending on fine-tuning mode
        if self.training_mode == "linear":
            # Freeze transformer layers and keep classifier unfrozen
            for name, param in self.net.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        elif self.training_mode == "lora":
            
            # Wrap in LoRA model
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                modules_to_save=["classifier"],
            )
            if self.optimizer not in ['svgd', "deep_ens", "fhbi"]:
                self.net = get_peft_model(self.net, config)
            else: #init multiple net @@ corresponding to different particles
                
                self.net = LoRA_ViT(num_particles=self.num_particles, vit_model=self.net, r=self.lora_r, alpha=self.lora_alpha, num_classes=self.n_classes)
                    
        elif self.training_mode == "block":
            config = AutoConfig.from_pretrained(model_path)
            config.image_size = self.image_size

            print('Number of Layers: ', config.num_hidden_layers)

            ckpt = self.net.state_dict()
            output, selected_layers = block_expansion(ckpt,
                                                      split,
                                                      config.num_hidden_layers)
            
            print('Selected Layers: ', selected_layers)

            config.num_hidden_layers += len(selected_layers)
            config.num_labels = self.n_classes

            self.net = AutoModelForImageClassification.from_config(config)
            self.net.load_state_dict(output)

            self.net.requires_grad_(False)
            for n, p in self.net.named_parameters():
                for idx in selected_layers:
                    if 'layer.' + str(idx) + '.' in n:
                        p.requires_grad_(True)

            self.net.classifier = torch.nn.Linear(config.hidden_size, self.n_classes)

        elif self.training_mode == "full":
            pass  # Keep all layers unfrozen
        else:
            raise ValueError(
                f"{self.training_mode} is not an available fine-tuning mode. Should be one of ['full', 'linear', 'lora']"
            )

        # Define metrics
        self.train_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                # "acc_top5": Accuracy(
                #     num_classes=self.n_classes,
                #     task="multiclass",
                #     top_k=min(5, self.n_classes),
                # ),
                #"ece": CalibrationError(num_classes=self.n_classes, norm='l1')
            }
        )
        self.val_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                # "acc_top5": Accuracy(
                #     num_classes=self.n_classes,
                #     task="multiclass",
                #     top_k=min(5, self.n_classes),
                # ),
                #"ece": CalibrationError(num_classes=self.n_classes, norm='l1')
            }
        )
        self.test_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                # "acc_top5": Accuracy(
                #     num_classes=self.n_classes,
                #     task="multiclass",
                #     top_k=min(5, self.n_classes),
                # ),
                #"ece": CalibrationError(num_classes=self.n_classes, norm='l1'),
                "stats": StatScores(
                    task="multiclass", average=None, num_classes=self.n_classes
                ),
                
            }
        )

        # Define loss
        self.loss_fn = SoftTargetCrossEntropy()

        # Define regularizers
        self.mixup = Mixup(
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            prob=self.mix_prob,
            label_smoothing=self.label_smoothing,
            num_classes=self.n_classes,
        )

        self.test_metric_outputs = []
        
        if self.optimizer in ['svgd', 'deep_ens', 'fhbi']:
            self.automatic_optimization = False

    def forward(self, x):
        if self.optimizer not in ['svgd', 'deep_ens', 'fhbi']:
            return self.net(x).logits
        else:
            res = self.net(x)
            return res
        
    def shared_step(self, batch, mode="train"):
        x, y = batch
        x, y = x.cuda(), y.cuda()

        if mode == "train":
            # Only converts targets to one-hot if no label smoothing, mixup or cutmix is set
            x, y = self.mixup(x, y)
        else:
            y = F.one_hot(y, num_classes=self.n_classes).float()

        
        if self.optimizer not in ['svgd', 'deep_ens', 'fhbi']:
            # Pass through network

            pred = self(x)
            loss = self.loss_fn(pred, y)
            # Get accuracy
            metrics = getattr(self, f"{mode}_metrics")(pred, y.argmax(1))
        elif self.optimizer == 'deep_ens' and mode == "train":
            
            scaled_epsilon = self.epsilon * (x.max() - x.min())

            # force inputs to require gradient
            inputs = x.clone()
            inputs.requires_grad = True
            
            # standard forwards pass
            pred = self(inputs)
                                
            pred_ = 0 #pred
            for j in range(self.num_particles):
                pred_ = pred_ + pred[j]
            pred_ = pred_/max(1, self.num_particles)
            loss = self.loss_fn(pred_, y)
            
            # now compute gradients wrt input
            self.optimizers().zero_grad()
            self.manual_backward(loss) #, retain_graph=True)
            # now compute sign of gradients
            inputs_grad = torch.sign(inputs.grad)

            # perturb inputs and use clamped output
            inputs_perturbed = torch.clamp(
                inputs + scaled_epsilon * inputs_grad, 0.0, 1.0
            ).detach()
            inputs.grad.zero_()
            
            inputs_all = torch.cat((inputs, inputs_perturbed), dim=0)
            outputs_all = self(inputs_all)

            # compute adversarial version of loss
            pred_p = 0 #pred
            for j in range(self.num_particles):
                pred_p = pred_p + outputs_all[j]
            pred_p = pred_p/max(1, self.num_particles)
            final_loss = (self.loss_fn(pred_p[:y.shape[0]], y) + self.loss_fn(pred_p[y.shape[0]:], y))/2.0
            
            loss = final_loss
            
            # Get accuracy
            metrics = getattr(self, f"{mode}_metrics")(pred_, y.argmax(1))

        else:
            pred = self(x)
            pred_ = 0 #pred
            for j in range(self.num_particles):
                pred_ = pred_ + pred[j]
            pred_ = pred_/max(1, self.num_particles)
            loss = self.loss_fn(pred_, y)
            
            # Get accuracy
            metrics = getattr(self, f"{mode}_metrics")(pred_, y.argmax(1))

        # Log
        self.log(f"{mode}_loss", loss.item(), on_epoch=True)
        for k, v in metrics.items():
            if len(v.size()) == 0:
                self.log(f"{mode}_{k.lower()}", v, on_epoch=True)

        if mode == "test":
            self.test_metric_outputs.append(metrics["stats"])
            
        return loss

    def training_step(self, batch, _):
        if self.optimizer == 'svgd':
            self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
            opt = self.optimizers()
            scheduler = self.lr_schedulers()
            
            torch.autograd.set_detect_anomaly(True)

            loss = self.shared_step(batch, "train")
            
            opt.zero_grad()
            self.manual_backward(loss)

            # for sam
            opt.step_()
            opt.zero_grad()
            
            scheduler.step()
            # return loss
        elif self.optimizer == 'fhbi':
            self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
            opt = self.optimizers()
            scheduler = self.lr_schedulers()
            
            torch.autograd.set_detect_anomaly(True)

            loss = self.shared_step(batch, "train")
            
            opt.zero_grad()
            self.manual_backward(loss)

            org_weight_tuple, kernel_tuple = opt.step1()
            loss = self.shared_step(batch, "train")
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step2(org_weight_tuple, kernel_tuple)
                    
            opt.zero_grad()
            
            scheduler.step()
            # return loss
        else:
            opt = self.optimizers()
            scheduler = self.lr_schedulers()
            loss = self.shared_step(batch, "train")
            
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            scheduler.step()
            
            self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
            # return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        val = self.shared_step(batch, "val")
        # self.test_step(batch, _)
        return val
    
    def on_validation_epoch_end(self):
        test_dataloader = self.trainer.datamodule.test_dataloader()
        for batch in test_dataloader:
            self.test_step(batch, 0)
            
    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        """Save per-class accuracies to csv"""
        # Aggregate all batch stats
        combined_stats = torch.sum(
            torch.stack(self.test_metric_outputs, dim=-1), dim=-1
        )

        # Calculate accuracy per class
        per_class_acc = []
        for tp, _, _, _, sup in combined_stats:
            acc = tp / sup
            per_class_acc.append((acc.item(), sup.item()))

        # Save to csv
        df = pd.DataFrame(per_class_acc, columns=["acc", "n"])
        df.to_csv("per-class-acc-test.csv")
        print("Saved per-class results in per-class-acc-test.csv")

    def configure_optimizers(self):
        # Initialize optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer in ["sgd", 'deep_ens']:
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "svgd":  #use Adam as the base optimizer by default @@        
            optimizer =  SVGD(
                param = self.net.parameters(), 
                rho=self.rho,
                sigma=self.sigma, 
                lr=self.lr, 
                betas=self.betas,
                weight_decay=self.weight_decay, 
                num_particles=self.num_particles, 
                train_module=self, 
                net=self.net, 
                )
        elif self.optimizer == "fhbi":  #use Adam as the base optimizer by default @@        
            optimizer =  FHBI(
                net=self.net,
                param = self.net.parameters(), 
                rho=self.rho,
                sigma=self.sigma, 
                lr=self.lr, 
                betas=self.betas,
                weight_decay=self.weight_decay, 
                num_particles=self.num_particles, 
                train_module=self, 
                base_optimizer_name=self.base_optimizer_name, 
                )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd', 'deepEns']"
            )

        # Initialize learning rate scheduler
        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                num_warmup_steps=self.warmup_steps,
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 0.99)
            
        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }