import cv2
import numpy as np

from local_recognizer import LocalRecognizer


def find_runs(mask):
    """返回布尔序列中连续为True的区间：[start, end)。"""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def merge_runs(runs, max_gap):
    """合并间隔较小的投影区间。"""
    merged = []

    for start, end in runs:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return merged


class VerticalOCR:
    def __init__(self):
        # 模型文件检查和加载由LocalRecognizer负责。
        self.engine = LocalRecognizer()

    def recognize(
        self,
        image,
        margin_ratio=0.02,
        column_gap_ratio=0.25,
        char_gap_ratio=0.16,
    ):
        """
        识别简单竖排古籍页面。

        阅读顺序：栏从右向左，每栏从上向下。
        坐标以输入的原始页面图片为基准。

        返回：
            text：按阅读顺序拼接的文字，栏之间使用换行符。
            tokens：各识别区域的文字、置信度、坐标和文本偏移。

        注意：
            字框来自投影切分，不是模型输出的字符级对齐结果。
        """
        if not 0 <= margin_ratio < 0.5:
            raise ValueError("margin_ratio必须在0至0.5之间，不包含0.5。")
        if column_gap_ratio < 0 or char_gap_ratio < 0:
            raise ValueError("间隙合并系数不能为负数。")

        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape

        mx = int(width * margin_ratio)
        my = int(height * margin_ratio)

        binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]

        # 忽略页面边缘，但不裁剪图片，不改变坐标系。
        if mx:
            binary[:, :mx] = 0
            binary[:, width - mx:] = 0

        if my:
            binary[:my, :] = 0
            binary[height - my:, :] = 0

        # 根据连通区域宽度估计横向合并阈值。
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        sizes = [
            int(stats[i, cv2.CC_STAT_WIDTH])
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= 8
            and 3 <= stats[i, cv2.CC_STAT_WIDTH] <= width * 0.1
            and 3 <= stats[i, cv2.CC_STAT_HEIGHT] <= height * 0.1
        ]

        typical_width = (
            float(np.median(sizes))
            if sizes
            else width / 40
        )

        # 沿x方向统计墨迹，寻找竖排文字栏。
        x_projection = np.count_nonzero(binary, axis=0)
        x_mask = x_projection > max(2, height * 0.003)

        columns = merge_runs(
            find_runs(x_mask),
            max(1, int(typical_width * column_gap_ratio)),
        )

        columns = [
            (x0, x1)
            for x0, x1 in columns
            if x1 - x0 >= max(4, typical_width * 0.4)
        ]

        boxes = []

        # columns原本从左向右排列，反转后按古籍顺序读取。
        for column_index, (x0, x1) in enumerate(reversed(columns)):
            column_width = x1 - x0

            # 对当前栏沿y方向投影，寻找单字区域。
            y_projection = np.count_nonzero(
                binary[:, x0:x1],
                axis=1,
            )
            y_mask = y_projection > max(
                1,
                column_width * 0.025,
            )

            rows = merge_runs(
                find_runs(y_mask),
                max(1, int(column_width * char_gap_ratio)),
            )

            for y0, y1 in rows:
                block_height = y1 - y0

                if block_height < column_width * 0.15:
                    continue

                # 对较长的粘连区域，按近似方形字高等分。
                # 这只是简单版式下的启发式处理。
                parts = (
                    max(1, int(round(block_height / column_width)))
                    if block_height > column_width * 1.6
                    else 1
                )

                edges = np.linspace(
                    y0,
                    y1,
                    parts + 1,
                ).astype(int)

                for index in range(parts):
                    top = int(edges[index])
                    bottom = int(edges[index + 1])
                    pad = 2

                    left = max(0, x0 - pad)
                    upper = max(0, top - pad)
                    right = min(width, x1 + pad)
                    lower = min(height, bottom + pad)

                    boxes.append({
                        "column": column_index,
                        "x": left,
                        "y": upper,
                        "w": right - left,
                        "h": lower - upper,
                    })

        # 空白页或未切分出文字时，返回空结果。
        if not boxes:
            return "", []

        # 裁切字图，并转换成识别器需要的OpenCV BGR格式。
        crops = []

        for box in boxes:
            x = box["x"]
            y = box["y"]
            w = box["w"]
            h = box["h"]

            crop = rgb[y:y + h, x:x + w]
            crop = cv2.copyMakeBorder(
                crop,
                6,
                6,
                6,
                6,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

            crops.append(
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            )

        # 批量识别；必须保持在recognize方法内部。
        results = []

        for start in range(0, len(crops), 32):
            batch = crops[start:start + 32]
            results.extend(self.engine.recognize(batch))

        if len(results) != len(boxes):
            raise RuntimeError(
                "OCR返回数量与切分区域数量不一致："
                f"{len(results)}个结果，{len(boxes)}个区域。"
            )

        text_parts = []
        tokens = []
        offset = 0
        previous_column = None

        for box, result in zip(boxes, results):
            recognized, confidence = result
            recognized = str(recognized).strip() or "□"

            if (
                previous_column is not None
                and box["column"] != previous_column
            ):
                text_parts.append("\n")
                offset += 1

            token = dict(box)
            token.update({
                "i": len(tokens),
                "text": recognized,
                "confidence": float(confidence),
                "start": offset,
                "end": offset + len(recognized),
            })

            tokens.append(token)
            text_parts.append(recognized)

            offset += len(recognized)
            previous_column = box["column"]

        return "".join(text_parts), tokens
