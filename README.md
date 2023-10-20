# MiniGPT-5: Where Vision and Language Collide in Magic! ✨📷📝

Welcome to the enchanting realm of MiniGPT-5, where the mystical fusion of vision and language produces pure magic! This extraordinary journey is guided by the brilliant conjurors [Kaizhi Zheng](https://kzzheng.github.io/) ✨, [Xuehai He](https://scholar.google.com/citations?user=kDzxOzUAAAAJ&hl=en) 🧙‍♂️, and [Xin Eric Wang](https://eric-xw.github.io/) 🧚‍♂️, hailing from the mystical grounds of the University of California, Santa Cruz 🎓.

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2310.02239)
![teaser](figs/teaser.png)

At MiniGPT-5, we've mastered the art of making the impossible possible. Our grand mission? Unleashing the power of computers to generate mesmerizing images and captivating narratives. Prepare to be spellbound! 🖼️➡️📖

## Unveiling the Sorcery of MiniGPT-5 ✨🔮

The heart of our magic lies in the mystical "generative vokens." These enchanting elements serve as the conduits that weave images and text into a harmonious dance of creativity 🪄💃📷. But that's not all! We've honed our model through two distinct stages, elevating it to even greater heights:

- Stage 1: We craft breathtaking images without the need for lengthy descriptions. Who has time for those, anyway? 📜❌
- Stage 2: We steer the model without the constraints of classifiers, empowering it to create even smarter and more brilliant images. 🧠🚀

Behold MiniGPT-5, our magnum opus, which triumphs in the MMDialog dataset and receives resounding applause in human evaluations for the VIST dataset. It stands as a benchmark to be reckoned with! 🏆👏

## The Spellbook: Model Architecture 📚🏰

Delve into the secrets of our magical realm with a glimpse into the blueprint of our enchanting model:

![arch](figs/structure.png)

## Let the Magic Unfold! 🎩🔥

Ready to embark on this mystical journey? Begin by conjuring the perfect environment for MiniGPT-5:

1. **Summon the Repository and Create a Magical Sanctuary**

   ```bash
   git clone https://github.com/eric-ai-lab/MiniGPT-5.git
   cd MiniGPT-5
   conda create -n minigpt5 python=3.9
   conda activate minigpt5
   pip install -r requirements.txt
   ```

2. **Channel the Powers of Pretrained Weights**

   Our model draws its strength from the venerable [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), which includes the legendary [Vicuna](https://github.com/lm-sys/FastChat) and [BLIP-2](https://github.com/salesforce/LAVIS). Acquire the sacred [Vicuna V0 7B](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) weights and embed them in the [model config file](minigpt4/configs/models/minigpt4.yaml#L16).

   As for the Pretrained MiniGPT-4 Aligned Checkpoint, we've already safeguarded it within our mystical archives.

3. **Retrieve the Magical MiniGPT-5 Checkpoints**

   To set the magic in motion, you must procure the dual-stage checkpoints. Always commence with the ethereal Stage 1 weights:

   - [Stage 1: CC3M](https://drive.google.com/file/d/1y-VUXubIzFe0iq5_CJUaE3HKhlrdn4n2/view?usp=sharing)
   - [Stage 2: VIST](https://drive.google.com/file/d/1rjTsKwF8_pqcNLbdZdurqZLSpKoo2K9F/view?usp=drive_link)
   - [Stage 2: MMDialog](https://drive.google.com/file/d/1ehyX8Ykn1pbU5J8yM47catSswId0m5FZ/view?usp=drive_link)

   Safeguard these artifacts within a sacred folder, one that we shall christen ***WEIGHT_FOLDER***.

### Witness the Magic Unveil! 🪄✨

We beckon you to partake in our mystical demonstrations and assessments. Here's how you can wield your magical wand:

- **Demonstration**: Invoke the powers of our [playground](examples/playground.py) to conjure multimodal wonders.

   ```bash
   cd examples
   export IS_STAGE2=True
   python3 playground.py --stage1_weight WEIGHT_FOLDER/stage1_cc3m.ckpt --test_weight WEIGHT_FOLDER/stage2_vist.ckpt
   ```

- **Assessment**: Our magical realm is attuned to three mystical datasets - [CC3M](https://ai.google.com/research/ConceptualCaptions/download), [VIST](https://visionandlanguage.net/VIST/), and [MMDialog](https://github.com/victorsungo/MMDialog). Prepare yourself for the art of measurement:

   **1. Stage 1 (CC3M) Assessment**

   For crafting images from textual cues:

   ```bash
   export IS_STAGE2=False
   export WEIGHTFOLDER=WEIGHT_FOLDER
   export DATAFOLDER=datasets/CC3M
   python3 train_eval.py --test_data_path cc3m_val.tsv --test_weight stage1_cc3m.ckpt --gpus 0
   ```

   Measure your magical achievements:

   ```bash
   export CC3M_FOLDER=datasets/CC3M
   python3 metric.py --test_weight stage1_cc3m.ckpt
   ```

   **2. Stage 2 (VIST) Assessment**

   Your model is now adept at conjuring both images and text. An awe-inspiring feat, indeed!

   ```bash
   export IS_STAGE2=True
   export WEIGHTFOLDER=WEIGHT_FOLDER
   export DATAFOLDER=datasets/VIST
   python3 train_eval.py --test_data_path val_cleaned.json --test_weight stage2_vist.ckpt --stage1_weight stage1_cc3m.ckpt --gpus 0
   ```

   Calculate more mystical metrics:

   ```bash
   python3 metric.py --test_weight stage2_vist.ckpt
   ```

   **3. Stage 2 (MMDialog) Assessment**

   For the creation of multimodal responses within conversational spells:

   ```bash
   export IS_STAGE2=True
   export WEIGHTFOLDER=WEIGHT_FOLDER
   export DATAFOLDER=datasets/MMDialog
   python3 train_eval.py --test_data_path test/test_conversations.txt --

test_weight stage2_mmdialog.ckpt --stage1_weight stage1_cc3m.ckpt --gpus 0
   ```

   Measure your enchanting abilities:

   ```bash
   python3 metric.py --test_weight stage2_mmdialog.ckpt
   ```

## The Art of Mastery ✨📜

If you aspire to become a true sorcerer, we welcome you to join us in crafting MiniGPT-5's magic. Learn the incantations for your training spells:

1. **Stage 1 Training**

   Acquire the CC3M dataset, prepare it, and embark on your initiation:

   ```bash
   export IS_STAGE2=False
   export WEIGHTFOLDER=WEIGHT_FOLDER
   export DATAFOLDER=datasets/CC3M
   python3 train_eval.py --is_training True --train_data_path cc3m_val.tsv --val_data_path cc3m_val.tsv --model_save_name stage1_cc3m_{epoch}-{step} --gpus 0
   ```

2. **Stage 2 Training**

   If you're ready for the most profound mysteries, acquire the VIST or MMDialog datasets and launch your second-stage training:

   - For VIST:

     ```bash
     export IS_STAGE2=True
     export WEIGHTFOLDER=WEIGHT_FOLDER
     export DATAFOLDER=datasets/VIST
     python3 train_eval.py --is_training True --train_data_path val_cleaned.json --val_data_path val_cleaned.json --stage1_weight stage1_cc3m.ckpt --model_save_name stage2_vist_{epoch}-{step} --gpus 0
     ```

## Citing Our Enchantments 📝👑

If you find MiniGPT-5's enchantments as captivating as we do, we invite you to acknowledge our sorcery in your mystical research:

```bibtex
@misc{zheng2023minigpt5,
      title={MiniGPT-5: Where Vision and Language Collide in Magic},
      author={Kaizhi Zheng and Xuehai He and Xin Eric Wang},
      year={2023},
      journal={arXiv preprint arXiv:2310.02239}
}
```

Let us unite to make the world a more magical place! 🌍🎩✨