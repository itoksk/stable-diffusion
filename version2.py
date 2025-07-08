#!/usr/bin/env python3
"""
Stable Diffusion 画像生成プログラム Version 2
複数のモデルから選択可能、各モデルの特徴説明付き
"""

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL
import torch
import datetime
import os
import random

# 利用可能なモデルのリスト（すべて無料・安全）
MODELS = {
    "1": {
        "name": "Anything V5",
        "id": "stablediffusionapi/anything-v5",
        "type": "SD1.5",
        "vae": "stabilityai/sd-vae-ft-mse",
        "category": "アニメ・イラスト",
        "description": "万能型アニメモデル。幅広いアニメスタイルに対応し、初心者にも使いやすい。",
        "sample_prompts": [
            "1girl, school uniform, cherry blossoms, smile",
            "fantasy knight, armor, castle background",
            "cute cat ears girl, magical girl outfit"
        ]
    },
    "2": {
        "name": "Realistic Vision V6.0",
        "id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "type": "SD1.5",
        "vae": "stabilityai/sd-vae-ft-mse-original",
        "category": "写実的・リアル",
        "description": "最高峰の写実的モデル。人物、動物、風景すべてでフォトリアルな画像を生成。",
        "sample_prompts": [
            "professional portrait photo of a woman, natural lighting, bokeh",
            "majestic lion in african savanna, golden hour, wildlife photography",
            "modern cityscape at night, neon lights, rain, cinematic"
        ]
    },
    "3": {
        "name": "Counterfeit V3.0",
        "id": "gsdf/Counterfeit-V3.0",
        "type": "SD1.5",
        "vae": "stabilityai/sd-vae-ft-mse",
        "category": "アニメ・イラスト",
        "description": "高品質なアニメイラスト特化。繊細で美しいキャラクター表現が得意。",
        "sample_prompts": [
            "1girl, detailed eyes, flowing hair, fantasy dress, magical aura",
            "anime style, vibrant colors, dynamic pose, action scene",
            "kawaii chibi character, pastel colors, heart decorations"
        ]
    },
    "4": {
        "name": "DreamShaper",
        "id": "Lykon/DreamShaper",
        "type": "SD1.5",
        "vae": "stabilityai/sd-vae-ft-mse",
        "category": "万能・アート",
        "description": "リアルからファンタジーまで幅広く対応。特にSF・サイバーパンク風景が得意。",
        "sample_prompts": [
            "cyberpunk city, neon lights, flying cars, rain",
            "ethereal fantasy landscape, floating islands, magical crystals",
            "steampunk inventor workshop, gears and machinery, vintage"
        ]
    },
    "5": {
        "name": "EpicRealism",
        "id": "emilianJR/epiCRealism",
        "type": "SD1.5",
        "vae": "stabilityai/sd-vae-ft-mse",
        "category": "写実的・リアル",
        "description": "自然な肌の質感と照明が特徴。人物写真に最適。",
        "sample_prompts": [
            "close-up portrait, natural skin texture, soft lighting",
            "candid street photography, natural expressions, urban setting",
            "fashion photography, elegant dress, studio lighting"
        ]
    }
}

class StableDiffusionGenerator:
    def __init__(self):
        self.pipe = None
        self.current_model = None
        
    def show_model_menu(self):
        """利用可能なモデルのメニューを表示"""
        print("\n" + "="*70)
        print("利用可能なモデル一覧")
        print("="*70)
        
        for key, model in MODELS.items():
            print(f"\n[{key}] {model['name']} ({model['category']})")
            print(f"    {model['description']}")
            print(f"    例: {model['sample_prompts'][0]}")
        
        print("\n" + "="*70)
        
    def load_model(self, model_key):
        """選択されたモデルを読み込む"""
        if model_key not in MODELS:
            raise ValueError(f"無効なモデル番号です: {model_key}")
        
        model_info = MODELS[model_key]
        print(f"\n{model_info['name']}を読み込んでいます...")
        
        # VAEの読み込み
        vae = AutoencoderKL.from_pretrained(model_info['vae'])
        
        # スケジューラーの設定
        if model_info['category'] == "アニメ・イラスト":
            scheduler = EulerDiscreteScheduler.from_pretrained(
                model_info['id'], 
                subfolder="scheduler"
            )
        else:
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_info['id'], 
                subfolder="scheduler"
            )
        
        # パイプラインの作成
        if model_info['type'] == "SD1.5":
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_info['id'],
                scheduler=scheduler,
                vae=vae,
                safety_checker=None,
                requires_safety_checker=False,
                custom_pipeline="lpw_stable_diffusion"
            )
        else:  # SDXL
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_info['id'],
                scheduler=scheduler,
                vae=vae,
                use_safetensors=True
            )
        
        # GPUに転送
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
            print("GPUを使用します")
        else:
            print("CPUを使用します（生成に時間がかかります）")
            
        self.current_model = model_info
        print(f"{model_info['name']}の読み込みが完了しました！")
        
    def generate_image(self, prompt, negative_prompt="", num_images=1, 
                      width=512, height=512, cfg_scale=7, steps=20, seed=-1):
        """画像を生成"""
        if self.pipe is None:
            raise RuntimeError("モデルが読み込まれていません")
        
        # 出力ディレクトリの作成
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n画像を生成中...")
        print(f"プロンプト: {prompt}")
        
        generated_images = []
        
        for i in range(num_images):
            # シード値の設定
            if seed == -1:
                current_seed = random.randint(0, 2147483647)
            else:
                current_seed = seed + i
                
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(current_seed)
            
            # 画像生成
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                generator=generator,
                guidance_scale=cfg_scale,
                num_inference_steps=steps
            ).images[0]
            
            # ファイル名の生成
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.current_model['name'].replace(' ', '_')}_{timestamp}_{current_seed}.png"
            filepath = os.path.join(output_dir, filename)
            
            # 画像を保存
            image.save(filepath)
            generated_images.append(filepath)
            print(f"画像を保存しました: {filepath}")
            
        return generated_images

def main():
    """メインプログラム"""
    print("\n🎨 Stable Diffusion 画像生成プログラム Version 2")
    print("複数のモデルから選択して画像を生成できます")
    
    generator = StableDiffusionGenerator()
    
    while True:
        # モデル選択
        generator.show_model_menu()
        model_choice = input("\nモデルを選択してください (番号を入力): ")
        
        try:
            generator.load_model(model_choice)
        except Exception as e:
            print(f"エラー: {e}")
            continue
            
        # サンプルプロンプトの表示
        model_info = MODELS[model_choice]
        print(f"\n💡 {model_info['name']}のサンプルプロンプト:")
        for i, prompt in enumerate(model_info['sample_prompts'], 1):
            print(f"   {i}. {prompt}")
        
        # 画像生成ループ
        while True:
            print("\n" + "-"*50)
            prompt = input("プロンプトを入力 (qでモデル選択に戻る): ")
            
            if prompt.lower() == 'q':
                break
                
            # 詳細設定
            use_default = input("デフォルト設定を使用しますか？ (y/n): ").lower() == 'y'
            
            if use_default:
                negative_prompt = "nsfw, low quality, blurry" if model_info['category'] == "写実的・リアル" else "nsfw"
                num_images = 1
                width = 512
                height = 512
                cfg_scale = 7
                steps = 20
                seed = -1
            else:
                negative_prompt = input("ネガティブプロンプト (Enter でスキップ): ")
                num_images = int(input("生成枚数 (デフォルト: 1): ") or "1")
                width = int(input("画像の幅 (デフォルト: 512): ") or "512")
                height = int(input("画像の高さ (デフォルト: 512): ") or "512")
                cfg_scale = float(input("CFG Scale (デフォルト: 7): ") or "7")
                steps = int(input("ステップ数 (デフォルト: 20): ") or "20")
                seed = int(input("シード値 (-1でランダム): ") or "-1")
            
            try:
                # 画像生成
                generated_files = generator.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images=num_images,
                    width=width,
                    height=height,
                    cfg_scale=cfg_scale,
                    steps=steps,
                    seed=seed
                )
                
                print(f"\n✅ {len(generated_files)}枚の画像を生成しました！")
                
            except Exception as e:
                print(f"\n❌ エラーが発生しました: {e}")
                
        # 別のモデルを試すか確認
        if input("\n別のモデルを試しますか？ (y/n): ").lower() != 'y':
            break
    
    print("\n👋 ご利用ありがとうございました！")

if __name__ == "__main__":
    main()