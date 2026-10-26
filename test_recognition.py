from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from local_recognizer import LocalRecognizer


BASE = Path(__file__).resolve().parent


def main():
    image_path = BASE / "test_line.png"

    if not image_path.is_file():
        raise FileNotFoundError(
            f"请先准备一张单行横排繁体图片：{image_path}"
        )

    with Image.open(image_path) as image:
        # 将透明背景合成白色，避免透明PNG转成黑底。
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb_image = Image.alpha_composite(white, rgba).convert("RGB")

    rgb = np.asarray(rgb_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    recognizer = LocalRecognizer()

    print("模型目录：", recognizer.model_dir)
    print("字典路径：", recognizer.dict_path)
    print("图片尺寸：", rgb_image.size)
    print("识别算法：", recognizer.engine.rec_algorithm)
    print("输入形状配置：", recognizer.engine.rec_image_shape)
    print("输出类别数：", recognizer.output_classes)

    # 不做投影切分，不旋转，不二值化。
    result = recognizer.recognize([bgr])
    text, confidence = result[0]

    print("\n原始识别文字：", repr(text))
    print("识别置信度：", confidence)

    # 保存实际送入识别器的图像，便于检查。
    output_path = BASE / "test_line_input.png"
    rgb_image.save(output_path)
    print("输入图像已保存：", output_path)


if __name__ == "__main__":
    main()
