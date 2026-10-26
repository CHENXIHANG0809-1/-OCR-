# local_recognizer.py
from pathlib import Path
from threading import Lock

import numpy as np

from paddleocr.tools.infer.predict_rec import TextRecognizer
from paddleocr.tools.infer.utility import init_args


BASE = Path(__file__).resolve().parent


class LocalRecognizer:
    """仅加载本地PP-OCRv3识别模型，不初始化检测器或方向分类器。"""

    def __init__(self):
        self.model_dir = BASE / "models" / "rec"
        self.dict_path = BASE / "models" / "chinese_cht_dict.txt"

        required = [
            self.model_dir / "inference.pdmodel",
            self.model_dir / "inference.pdiparams",
            self.dict_path,
        ]

        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"缺少文件：{path}")
            if path.stat().st_size == 0:
                raise ValueError(f"文件为空：{path}")

        # 字典顺序是类别编号的一部分，不能排序、去重或自行增删。
        with self.dict_path.open("r", encoding="utf-8") as file:
            entries = [line.rstrip("\r\n") for line in file]

        if not entries or any(entry == "" for entry in entries):
            raise ValueError("字典为空或包含空行，请使用原始配套字典。")

        if entries[0].startswith("\ufeff"):
            raise ValueError("字典含UTF-8 BOM，请使用原始无BOM字典。")

        args = init_args().parse_args([])

        args.use_gpu = False
        args.use_onnx = False
        args.enable_mkldnn = False
        args.cpu_threads = 4

        args.rec_model_dir = str(self.model_dir)
        args.rec_char_dict_path = str(self.dict_path)

        # PP-OCRv3常规繁体识别模型采用SVTR_LCNet推理路径。
        args.rec_algorithm = "SVTR_LCNet"
        args.rec_image_shape = "3, 48, 320"
        args.rec_batch_num = 16
        args.use_space_char = True

        self.engine = TextRecognizer(args)
        self.lock = Lock()

        # 初始化后运行一次，尽早检查推理和字典维度问题。
        self._check_output_classes()

    def _check_output_classes(self):
        """
        直接运行底层预测器，并在输出内存被回收前读取结果。

        这里检查模型能否推理，以及CTC输出类别数是否与字典相容；
        不检查实际文字识别准确率。
        """
        # PP-OCRv3识别输入：NCHW，float32。
        # 全零是合法的归一化测试输入，仅用于检查推理和输出形状。
        dummy = np.zeros((1, 3, 48, 320), dtype=np.float32)

        with self.lock:
            self.engine.input_tensor.copy_from_cpu(dummy)
            self.engine.predictor.run()

            # 必须在识别器高层调用回收内存之前复制输出。
            output = self.engine.output_tensors[0].copy_to_cpu()

        if output.ndim != 3:
            raise RuntimeError(
                "预期CTC输出维度为[N,T,C]，"
                f"实际输出形状为：{output.shape}"
            )

        if not np.isfinite(output).all():
            raise RuntimeError("模型输出含NaN或无穷值，请检查模型和运行环境。")

        expected = len(self.engine.postprocess_op.character)
        actual = int(output.shape[-1])

        if actual != expected:
            raise RuntimeError(
                f"模型输出类别数为{actual}，"
                f"解码字典类别数为{expected}。"
                "请检查字典是否为该模型的原始配套字典，"
                "以及use_space_char设置是否与模型一致。"
                "不要通过随意增删字典字符来凑类别数量。"
            )

        self.output_classes = actual
    def recognize(self, images):
        """
        images：OpenCV BGR图像列表。
        返回：[(text, confidence), ...]，与输入顺序一致。
        """
        if not images:
            return []

        for index, image in enumerate(images):
            if (
                not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[2] != 3
                or image.size == 0
            ):
                raise ValueError(f"第{index}张裁切图片格式无效。")

        # Streamlit缓存的实例可能被多个会话共享，避免同时调用预测器。
        with self.lock:
            results, _elapsed = self.engine(images)

        if len(results) != len(images):
            raise RuntimeError(
                f"输入{len(images)}张图片，但返回{len(results)}个结果。"
            )

        return [(str(text), float(score)) for text, score in results]
