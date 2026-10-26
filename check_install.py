# check_install.py
import argparse
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import paddle
from PIL import Image, ImageDraw

from local_recognizer import LocalRecognizer


BASE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crop",
        help="单个正向汉字的裁切图，用于检查识别器",
    )
    parser.add_argument(
        "--page",
        help="完整竖排页面图片，用于检查切分、坐标和OCR",
    )
    args = parser.parse_args()

    print("Python:", sys.version)
    print("平台:", platform.platform())
    print("Paddle:", paddle.__version__)

    paths = [
        BASE / "models" / "rec" / "inference.pdmodel",
        BASE / "models" / "rec" / "inference.pdiparams",
        BASE / "models" / "chinese_cht_dict.txt",
    ]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"\n文件：{path.name}")
        print(f"大小：{path.stat().st_size:,} bytes")
        print(f"SHA256：{sha256(path)}")

    recognizer = LocalRecognizer()
    print("\n模型加载和推理成功。")
    print("模型输出类别数：", recognizer.output_classes)
    print("注意：维度匹配不等于字典顺序已验证。")

    if args.crop:
        with Image.open(args.crop) as image:
            rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()

        result = recognizer.recognize([bgr])
        print("\n单字／裁切图识别结果：", result)

    if args.page:
        from ocr_engine import VerticalOCR

        with Image.open(args.page) as source:
            image = source.convert("RGB")

        engine = VerticalOCR()
        text, tokens = engine.recognize(image)

        if not tokens:
            raise AssertionError("没有切分出文字区域。")

        for index, token in enumerate(tokens):
            assert token["i"] == index
            assert token["w"] > 0 and token["h"] > 0
            assert 0 <= token["x"] < image.width
            assert 0 <= token["y"] < image.height
            assert token["x"] + token["w"] <= image.width
            assert token["y"] + token["h"] <= image.height
            assert text[token["start"]:token["end"]] == token["text"]
            assert np.isfinite(token["confidence"])

        output = BASE / "debug_output"
        output.mkdir(exist_ok=True)

        (output / "page.txt").write_text(text, encoding="utf-8")
        (output / "tokens.json").write_text(
            json.dumps(tokens, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for token in tokens:
            x, y = token["x"], token["y"]
            draw.rectangle(
                (x, y, x + token["w"] - 1, y + token["h"] - 1),
                outline="red",
                width=2,
            )
            draw.text((x, y), str(token["i"]), fill="blue")

        annotated.save(output / "boxes.png")

        print(f"\n整页检查通过，共{len(tokens)}个文字区域。")
        print(f"调试结果保存在：{output}")
        print("请人工检查boxes.png与page.txt。")

    print("\n检查结束：未发现本次检查覆盖的运行错误。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
