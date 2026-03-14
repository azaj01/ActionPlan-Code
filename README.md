# ActionPlan: Future-Aware Streaming Motion Synthesis via Frame-Level Action Planning

<p align="center">
  <a href="https://coral79.github.io/ActionPlan/"><b>[🌐 Project Page]</b></a>
  <a href="#"><b>[📄 arXiv]</b></a>
</p>

This is the official repository for **ActionPlan**.

---

## 🚀 News
- **[March 2026]** Our paper has been submitted to arXiv!
- **[Coming Very Soon]** Inference code for the online demo.

## 📦 Release Plan
- [ ] **Inference Code**: Run the real-time streaming demo yourself.
- [ ] **Model Weights**: Pre-trained checkpoints for ActionPlan.
- [ ] **Training Code**: Full training pipeline.

## 📄 Abstract
We present **ActionPlan**, a unified motion diffusion framework that bridges real-time streaming with high-quality offline generation within a single model. The core idea is to introduce a *per-frame action plan*: the model predicts frame-level text latents that act as dense semantic anchors throughout denoising, and uses them to denoise the full motion sequence with combined semantic and motion cues.

To support this structured workflow, we design latent-specific diffusion steps, allowing each motion latent to be denoised independently and sampled in flexible orders at inference. As a result, ActionPlan can run in a history-conditioned, future-aware mode for real-time streaming, while also supporting high-quality offline generation.

The same mechanism further enables zero-shot motion editing and in-betweening without additional models. Experiments demonstrate that our real-time streaming is **5.25× faster** while achieving **18% motion quality improvement** over the best previous method in terms of FID.

## ✍️ Citation
If you find our work or code useful for your research, please consider citing:

```bibtex
@article{nazarenus2026actionplan,
  title   = {{ActionPlan}: Future-Aware Streaming Motion Synthesis via Frame-Level Action Planning},
  author  = {Nazarenus, Eric and Li, Chuqiao and He, Yannan and Xie, Xianghui and Lenssen, Jan Eric and Pons-Moll, Gerard},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## 👀 You Might Also Like

Also check out our CVPR 2026 paper 🧟 **[FrankenMotion](https://coral79.github.io/frankenmotion/)** — part-level human motion generation and composition with fine-grained spatial and temporal control.

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Coral79/ActionPlan-Code&type=Date)](https://star-history.com/#Coral79/ActionPlan-Code&Date)
