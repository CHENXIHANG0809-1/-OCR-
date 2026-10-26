import base64
import html
import io
import json
import sqlite3
import uuid
from pathlib import Path

import fitz
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from ocr_engine import VerticalOCR


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
FILES = DATA / "documents"
FILES.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "database.sqlite"


def connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    """
    创建基础数据表和人工校对文本表。

    如果将来使用的是已经存在的旧数据库，只会新增表，
    不会删除原有文献、页面和OCR结果。
    """
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                original_path TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                tokens_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'pending',
                error TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(id),
                UNIQUE(document_id, page_number)
            );

            CREATE INDEX IF NOT EXISTS idx_pages_document
            ON pages(document_id, page_number);

            CREATE TABLE IF NOT EXISTS manual_transcriptions (
                page_id INTEGER PRIMARY KEY,
                text TEXT NOT NULL DEFAULT '',
                source_filename TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(page_id) REFERENCES pages(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_manual_text
            ON manual_transcriptions(text);
            """
        )


def query(sql, parameters=()):
    with connect() as connection:
        return [
            dict(row)
            for row in connection.execute(sql, parameters).fetchall()
        ]


def execute(sql, parameters=()):
    with connect() as connection:
        connection.execute(sql, parameters)


def import_file(upload):
    """
    导入原始图片或PDF，并将每一页保存为PNG。
    """
    suffix = Path(upload.name).suffix.lower()

    if suffix not in {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }:
        raise ValueError("不支持的文件格式。")

    document_id = uuid.uuid4().hex
    directory = FILES / document_id
    directory.mkdir(parents=True, exist_ok=False)

    original = directory / ("original" + suffix)
    original.write_bytes(upload.getvalue())

    records = []

    def save_page(image, number):
        image = image.convert("RGB")
        destination = directory / f"page_{number:04d}.png"
        image.save(destination)

        records.append(
            (
                document_id,
                number,
                str(destination.relative_to(DATA)),
                image.width,
                image.height,
            )
        )

    if suffix == ".pdf":
        with fitz.open(original) as pdf:
            if pdf.needs_pass:
                raise ValueError("暂不支持加密PDF。")

            for index, pdf_page in enumerate(pdf):
                pixmap = pdf_page.get_pixmap(
                    dpi=200,
                    colorspace=fitz.csRGB,
                    alpha=False,
                )

                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )

                save_page(image, index + 1)

    else:
        with Image.open(original) as image:
            frame_count = getattr(image, "n_frames", 1)

            for index in range(frame_count):
                image.seek(index)
                save_page(image.copy(), index + 1)

    if not records:
        raise ValueError("文件中没有可导入页面。")

    with connect() as connection:
        connection.execute(
            """
            INSERT INTO documents(id, title, original_path)
            VALUES (?, ?, ?)
            """,
            (
                document_id,
                Path(upload.name).stem,
                str(original.relative_to(DATA)),
            ),
        )

        connection.executemany(
            """
            INSERT INTO pages(
                document_id,
                page_number,
                image_path,
                width,
                height
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )

    return document_id


def read_manual_file(upload):
    """
    读取TXT或DOCX人工校对文件。

    DOCX读取段落和表格中的文字；
    不直接把Word分页当成扫描页分页。
    """
    suffix = Path(upload.name).suffix.lower()
    data = upload.getvalue()

    if suffix == ".txt":
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue

        raise ValueError("TXT文件无法按UTF-8或GB18030读取。")

    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as error:
            raise RuntimeError(
                "读取DOCX需要安装python-docx："
                "python -m pip install python-docx"
            ) from error

        document = Document(io.BytesIO(data))
        parts = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                parts.append(text)

        for table in document.tables:
            for row in table.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                ]
                row_text = "    ".join(
                    item for item in cells if item
                )
                if row_text:
                    parts.append(row_text)

        return "\n\n".join(parts)

    raise ValueError("人工校对文件只支持TXT和DOCX。")


def save_manual_text(page_id, text, source_filename=None):
    """
    保存当前页面的人工校对文本。

    使用INSERT OR REPLACE，确保每页只有一条当前文本。
    """
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO manual_transcriptions(
                page_id,
                text,
                source_filename,
                updated_at
            )
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(page_id) DO UPDATE SET
                text=excluded.text,
                source_filename=excluded.source_filename,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                page_id,
                text,
                source_filename,
            ),
        )


def get_manual_text(page_id):
    rows = query(
        """
        SELECT text, source_filename, updated_at
        FROM manual_transcriptions
        WHERE page_id=?
        """,
        (page_id,),
    )

    if not rows:
        return {
            "text": "",
            "source_filename": "",
            "updated_at": "",
        }

    return rows[0]


def compact_text(text):
    """
    去除空白字符，但保留其对应的原始文本偏移。
    """
    characters = []
    offsets = []

    for index, character in enumerate(text):
        if not character.isspace():
            characters.append(character)
            offsets.append(index)

    return "".join(characters), offsets


def find_text_matches(text, keyword):
    """
    返回文本中所有匹配位置。
    """
    needle, _ = compact_text(keyword)

    if not needle:
        return []

    searchable, offsets = compact_text(text)
    matches = []
    position = 0

    while True:
        found = searchable.find(needle, position)

        if found < 0:
            break

        start = offsets[found]
        end = offsets[found + len(needle) - 1] + 1

        matches.append((start, end))
        position = found + 1

    return matches


def make_snippet(text, start, end):
    return text[
        max(0, start - 18):min(len(text), end + 18)
    ].replace("\n", " ")


def search_pages(keyword, source="两者", limit=100):
    """
    搜索OCR文本、人工校对文本，或两者。

    返回结果中source字段表示命中来源：
        OCR
        人工校对
    """
    if not keyword.strip():
        return []

    rows = query(
        """
        SELECT
            p.id,
            p.document_id,
            p.page_number,
            p.text AS ocr_text,
            d.title,
            COALESCE(m.text, '') AS manual_text
        FROM pages p
        JOIN documents d ON d.id=p.document_id
        LEFT JOIN manual_transcriptions m
            ON m.page_id=p.id
        WHERE p.status='done'
           OR COALESCE(m.text, '') <> ''
        ORDER BY d.created_at DESC, p.page_number
        """
    )

    matches = []

    for row in rows:
        sources = []

        if source in {"OCR结果", "两者"}:
            sources.append(
                (
                    "OCR",
                    row["ocr_text"] or "",
                )
            )

        if source in {"人工校对", "两者"}:
            sources.append(
                (
                    "人工校对",
                    row["manual_text"] or "",
                )
            )

        for source_name, text in sources:
            for start, end in find_text_matches(text, keyword):
                matches.append(
                    {
                        "id": row["id"],
                        "document_id": row["document_id"],
                        "page_number": row["page_number"],
                        "title": row["title"],
                        "source": source_name,
                        "text": text,
                        "start": start,
                        "end": end,
                        "snippet": make_snippet(text, start, end),
                    }
                )

                if len(matches) >= limit:
                    return matches

    return matches


def highlight_manual_text(text, hit):
    """
    对人工校对文本的检索命中部分进行高亮。
    """
    if not text:
        return "本页尚未录入人工校对文本。"

    if not hit:
        return html.escape(text).replace("\n", "<br>")

    start, end = hit
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))

    before = html.escape(text[:start])
    matched = html.escape(text[start:end])
    after = html.escape(text[end:])

    return (
        before
        + f'<mark class="manual-hit">{matched}</mark>'
        + after
    ).replace("\n", "<br>")


def render_reader(page, manual_text="", hit=None):
    """
    三栏阅读器：

    左侧：原始图像
    中间：OCR文字
    右侧：人工校对文字
    """
    tokens = json.loads(page["tokens_json"])

    hit_source = None
    hit_start = None
    hit_end = None

    if hit and hit[0] == page["id"]:
        hit_source = hit[3]
        hit_start = hit[1]
        hit_end = hit[2]

    with Image.open(DATA / page["image_path"]) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=88,
        )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    fragments = []
    previous_column = None

    for token in tokens:
        if (
            previous_column is not None
            and token.get("column") != previous_column
        ):
            fragments.append("<br>")

        classes = ["token"]

        if float(token.get("confidence", 0)) < 0.6:
            classes.append("low")

        if (
            hit_source == "OCR"
            and hit_start is not None
            and token["start"] < hit_end
            and token["end"] > hit_start
        ):
            classes.append("hit")

        token_text = html.escape(str(token.get("text", "")))

        fragments.append(
            f'<span class="{" ".join(classes)}" '
            f'data-i="{token["i"]}" '
            f'title="置信度 {float(token.get("confidence", 0)):.2f}">'
            f'{token_text}</span>'
        )

        previous_column = token.get("column")

    if hit_source == "人工校对":
        manual_html = highlight_manual_text(
            manual_text,
            (hit_start, hit_end),
        )
    else:
        manual_html = (
            html.escape(manual_text).replace("\n", "<br>")
            if manual_text
            else "本页尚未录入人工校对文本。"
        )

    token_json = json.dumps(
        tokens,
        ensure_ascii=True,
    ).replace("<", "\\u003c")

    template = r"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
    <meta charset="UTF-8">
    <style>
      body {
        margin: 0;
        font-family: serif;
        color: #222;
      }

      .toolbar {
        margin-bottom: 10px;
        font-family: sans-serif;
        font-size: 14px;
      }

      .layout {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 12px;
      }

      .pane {
        height: 680px;
        overflow: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
        background: #fafafa;
      }

      #stage {
        position: relative;
        width: 100%;
      }

      #scan {
        display: block;
        width: 100%;
        height: auto;
      }

      #overlay {
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
      }

      #ocrText,
      #manualText {
        padding: 20px;
        font-size: 22px;
        line-height: 1.9;
        overflow-wrap: anywhere;
        white-space: normal;
      }

      #manualText {
        background: #fffdf5;
      }

      .token {
        cursor: pointer;
      }

      .low {
        text-decoration: underline dotted #d98200;
      }

      .hit {
        background: #b8e5ff;
      }

      .selected {
        background: #ffc777;
      }

      .manual-hit {
        background: #ffca75;
        padding: 1px 2px;
      }

      button {
        margin-left: 8px;
      }

      .title {
        font-weight: bold;
        margin-bottom: 12px;
        color: #555;
      }
    </style>
    </head>

    <body>
      <div class="toolbar">
        左：原始扫描页　
        中：OCR结果　
        右：人工校对结果。
        OCR支持字框定位；人工校对结果当前按页面定位。
        <button id="minus">缩小</button>
        <button id="plus">放大</button>
      </div>

      <div class="layout">
        <div class="pane" id="imagePane">
          <div id="stage">
            <img id="scan" src="data:image/jpeg;base64,__IMAGE__">
            <svg id="overlay"
                 viewBox="0 0 __WIDTH__ __HEIGHT__"></svg>
          </div>
        </div>

        <div class="pane" id="ocrPane">
          <div class="title">OCR识别结果</div>
          <div id="ocrText">
            __OCR_TEXT__
          </div>
        </div>

        <div class="pane" id="manualPane">
          <div class="title">人工校对结果</div>
          <div id="manualText">
            __MANUAL_TEXT__
          </div>
        </div>
      </div>

    <script>
      const tokens = __TOKENS__;
      const svg = document.getElementById("overlay");
      const stage = document.getElementById("stage");
      const ocrText = document.getElementById("ocrText");
      const spans = Array.from(
        ocrText.querySelectorAll(".token")
      );
      const imagePane = document.getElementById("imagePane");
      const baseHits = spans.filter(
        s => s.classList.contains("hit")
      );
      let zoom = 100;

      function draw(elements, color, locate) {
        svg.replaceChildren();

        const selected = [...new Set(
          elements.map(e => Number(e.dataset.i))
        )]
          .map(i => tokens[i])
          .filter(Boolean);

        for (const token of selected) {
          const rect = document.createElementNS(
            "http://www.w3.org/2000/svg",
            "rect"
          );

          rect.setAttribute("x", token.x);
          rect.setAttribute("y", token.y);
          rect.setAttribute("width", token.w);
          rect.setAttribute("height", token.h);
          rect.setAttribute("fill", color);
          rect.setAttribute("fill-opacity", "0.35");
          rect.setAttribute("stroke", color);
          rect.setAttribute("stroke-width", "2");

          svg.appendChild(rect);
        }

        if (locate && selected.length) {
          const first = selected[0];
          const scale = stage.clientWidth / __WIDTH__;

          imagePane.scrollTo({
            left: Math.max(
              0,
              first.x * scale - imagePane.clientWidth / 2
            ),
            top: Math.max(
              0,
              first.y * scale - imagePane.clientHeight / 3
            ),
            behavior: "smooth"
          });
        }
      }

      function choose(elements) {
        spans.forEach(s => s.classList.remove("selected"));
        elements.forEach(s => s.classList.add("selected"));

        draw(
          elements.length ? elements : baseHits,
          elements.length ? "#ed8b16" : "#138ac7",
          true
        );
      }

      document.addEventListener("selectionchange", () => {
        const selection = window.getSelection();

        if (!selection.rangeCount || selection.isCollapsed) {
          return;
        }

        const range = selection.getRangeAt(0);

        const selected = spans.filter(span => {
          const node = span.firstChild;

          if (!node || !range.intersectsNode(node)) {
            return false;
          }

          if (
            range.startContainer === node
            && range.startOffset >= node.length
          ) {
            return false;
          }

          if (
            range.endContainer === node
            && range.endOffset === 0
          ) {
            return false;
          }

          return true;
        });

        if (selected.length) {
          choose(selected);
        }
      });

      ocrText.addEventListener("click", event => {
        if (!window.getSelection().isCollapsed) {
          return;
        }

        const span = event.target.closest(".token");

        if (span) {
          choose([span]);
        }
      });

      document.getElementById("plus").onclick = () => {
        zoom = Math.min(300, zoom + 25);
        stage.style.width = zoom + "%";
      };

      document.getElementById("minus").onclick = () => {
        zoom = Math.max(100, zoom - 25);
        stage.style.width = zoom + "%";
      };

      function initialHighlight() {
        draw(baseHits, "#138ac7", true);

        if (baseHits.length) {
          baseHits[0].scrollIntoView({
            block: "center"
          });
        }
      }

      const scan = document.getElementById("scan");

      if (scan.complete) {
        initialHighlight();
      } else {
        scan.onload = initialHighlight;
      }
    </script>
    </body>
    </html>
    """

    replacements = {
        "__IMAGE__": encoded,
        "__WIDTH__": str(page["width"]),
        "__HEIGHT__": str(page["height"]),
        "__OCR_TEXT__": "".join(fragments)
        or "本页尚无可显示的OCR结果。",
        "__MANUAL_TEXT__": manual_html,
        "__TOKENS__": token_json,
    }

    for key, value in replacements.items():
        template = template.replace(key, value)

    components.html(
        template,
        height=745,
        scrolling=False,
    )


init_db()

st.set_page_config(
    page_title="本地古籍OCR",
    layout="wide",
)

st.title("本地古籍OCR与人工校对")
st.caption(
    "左侧原始图像 · 中间OCR结果 · 右侧人工校对结果 · "
    "全部数据保存在本机"
)


with st.sidebar:
    st.header("导入文献")

    uploaded = st.file_uploader(
        "选择图片或PDF",
        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff",
        ],
        key="source_upload",
    )

    if st.button(
        "导入原始文献",
        disabled=uploaded is None,
        use_container_width=True,
    ):
        try:
            with st.spinner("正在导入并生成页面……"):
                document_id = import_file(uploaded)

            st.session_state["document_id"] = document_id
            st.session_state.pop("page_id", None)
            st.session_state.pop("hit", None)
            st.session_state.pop("search_results", None)
            st.session_state.pop("searched", None)

            st.success("原始文献导入完成。")

        except Exception as error:
            st.error(f"导入失败：{error}")


documents = query(
    """
    SELECT *
    FROM documents
    ORDER BY created_at DESC
    """
)

if not documents:
    st.info("请先在左侧导入图片或PDF原始文献。")
    st.stop()

document_map = {
    document["id"]: document
    for document in documents
}

if st.session_state.get("document_id") not in document_map:
    st.session_state["document_id"] = documents[0]["id"]


with st.sidebar:
    st.header("检索")

    search_source = st.selectbox(
        "检索范围",
        options=["人工校对", "OCR结果", "两者"],
        index=0,
    )

    keyword = st.text_input(
        "输入词句；检索时忽略空白和换栏",
    )

    if st.button(
        "搜索",
        disabled=not keyword.strip(),
        use_container_width=True,
    ):
        st.session_state["search_results"] = search_pages(
            keyword,
            source=search_source,
        )
        st.session_state["searched"] = True

    results = st.session_state.get("search_results", [])

    if st.session_state.get("searched"):
        st.caption(
            f"在{search_source}中找到{len(results)}处，最多显示100处。"
        )

    for index, result in enumerate(results):
        label = (
            f'{result["title"]} · 第{result["page_number"]}页'
            f' · {result["source"]}\n'
            f'{result["snippet"]}'
        )

        if st.button(
            label,
            key=f"search_result_{index}",
            use_container_width=True,
        ):
            st.session_state["document_id"] = result["document_id"]
            st.session_state["page_id"] = result["id"]
            st.session_state["hit"] = (
                result["id"],
                result["start"],
                result["end"],
                result["source"],
            )


document_id = st.selectbox(
    "当前文献",
    options=list(document_map),
    format_func=lambda value: document_map[value]["title"],
    key="document_id",
)


pages = query(
    """
    SELECT *
    FROM pages
    WHERE document_id=?
    ORDER BY page_number
    """,
    (document_id,),
)

page_map = {
    page["id"]: page
    for page in pages
}

if st.session_state.get("page_id") not in page_map:
    st.session_state["page_id"] = pages[0]["id"]


page_id = st.selectbox(
    "当前页面",
    options=list(page_map),
    format_func=lambda value: (
        f'第 {page_map[value]["page_number"]} 页'
        f' · {page_map[value]["status"]}'
    ),
    key="page_id",
)

page = page_map[page_id]
manual_record = get_manual_text(page_id)
manual_text = manual_record["text"]


with st.sidebar:
    st.header("人工校对")

    manual_upload = st.file_uploader(
        "导入TXT或DOCX人工校对文件",
        type=["txt", "docx"],
        key=f"manual_upload_{page_id}",
    )

    if st.button(
        "导入并覆盖当前页人工文本",
        disabled=manual_upload is None,
        use_container_width=True,
    ):
        try:
            imported_text = read_manual_file(manual_upload)

            if not imported_text.strip():
                raise ValueError("文件中没有可保存的文字。")

            save_manual_text(
                page_id=page_id,
                text=imported_text,
                source_filename=manual_upload.name,
            )

            st.session_state[
                f"manual_editor_{page_id}"
            ] = imported_text

            st.success(
                f"已将{manual_upload.name}绑定到当前第"
                f"{page['page_number']}页。"
            )

            st.rerun()

        except Exception as error:
            st.error(f"人工文本导入失败：{error}")


manual_editor_key = f"manual_editor_{page_id}"

if manual_editor_key not in st.session_state:
    st.session_state[manual_editor_key] = manual_text

st.subheader("当前页人工校对文本")

edited_manual_text = st.text_area(
    "可以直接粘贴或修改人工校对结果。保存后将绑定到当前页面。",
    key=manual_editor_key,
    height=260,
)

save_col, clear_col = st.columns(2)

with save_col:
    if st.button(
        "保存人工校对文本",
        use_container_width=True,
    ):
        save_manual_text(
            page_id=page_id,
            text=edited_manual_text,
            source_filename=manual_record["source_filename"]
            or "网页编辑",
        )

        st.success("人工校对文本已保存。")
        st.rerun()

with clear_col:
    if st.button(
        "清空当前页人工文本",
        use_container_width=True,
    ):
        execute(
            """
            DELETE FROM manual_transcriptions
            WHERE page_id=?
            """,
            (page_id,),
        )

        st.session_state[manual_editor_key] = ""
        st.success("当前页人工文本已清空。")
        st.rerun()


with st.expander("OCR切分参数"):
    st.caption(
        "参数只影响OCR识别，不会影响人工校对文本。"
    )

    margin = st.slider(
        "忽略页面边缘比例",
        0.0,
        0.15,
        0.02,
        0.005,
    )

    column_gap = st.slider(
        "栏内部横向间隙合并系数",
        0.05,
        1.0,
        0.25,
        0.05,
    )

    char_gap = st.slider(
        "字内部纵向间隙合并系数",
        0.02,
        0.5,
        0.16,
        0.02,
    )


left, right = st.columns(2)

run_current = left.button(
    "识别／重新识别当前页",
    use_container_width=True,
)

run_pending = right.button(
    "识别本书未完成页面",
    use_container_width=True,
)


if run_current or run_pending:
    targets = (
        [page]
        if run_current
        else [
            item
            for item in pages
            if item["status"] != "done"
        ]
    )

    if not targets:
        st.info("本书所有页面都已识别。")

    else:
        try:
            engine = get_ocr()
        except Exception as error:
            st.error(f"OCR初始化失败：{error}")
            st.stop()

        progress = st.progress(0)
        errors = []

        for index, target in enumerate(targets):
            try:
                with Image.open(
                    DATA / target["image_path"]
                ) as image:
                    recognized_text, tokens = engine.recognize(
                        image,
                        margin_ratio=margin,
                        column_gap_ratio=column_gap,
                        char_gap_ratio=char_gap,
                    )

                if not tokens:
                    raise RuntimeError(
                        "没有切分出文字，请检查图片或调整参数。"
                    )

                with connect() as connection:
                    connection.execute(
                        """
                        UPDATE pages
                        SET text=?,
                            tokens_json=?,
                            status='done',
                            error=NULL
                        WHERE id=?
                        """,
                        (
                            recognized_text,
                            json.dumps(
                                tokens,
                                ensure_ascii=False,
                            ),
                            target["id"],
                        ),
                    )

            except Exception as error:
                errors.append(
                    f'第{target["page_number"]}页：{error}'
                )

                with connect() as connection:
                    connection.execute(
                        """
                        UPDATE pages
                        SET status='failed',
                            error=?
                        WHERE id=?
                        """,
                        (
                            str(error),
                            target["id"],
                        ),
                    )

            progress.progress(
                (index + 1) / len(targets)
            )

        st.session_state["last_errors"] = errors
        st.session_state.pop("search_results", None)
        st.session_state.pop("searched", None)
        st.session_state.pop("hit", None)

        st.rerun()


for error in st.session_state.pop("last_errors", []):
    st.error(error)


if page["status"] == "failed":
    st.error(page["error"])


hit = st.session_state.get("hit")
hit_for_page = (
    hit
    if hit and hit[0] == page_id
    else None
)

render_reader(
    page=page,
    manual_text=manual_text,
    hit=hit_for_page,
)


download_col1, download_col2 = st.columns(2)

with download_col1:
    if page["text"]:
        st.download_button(
            "导出当前页OCR文本",
            data=page["text"].encode("utf-8"),
            file_name=(
                f'page_{page["page_number"]:04d}_ocr.txt'
            ),
            mime="text/plain",
            use_container_width=True,
        )

with download_col2:
    if manual_text:
        st.download_button(
            "导出当前页人工校对文本",
            data=manual_text.encode("utf-8"),
            file_name=(
                f'page_{page["page_number"]:04d}_manual.txt'
            ),
            mime="text/plain",
            use_container_width=True,
        )


if page["text"]:
    with st.expander("查看OCR坐标数据"):
        st.json(json.loads(page["tokens_json"]))

if manual_record["source_filename"]:
    st.caption(
        f'人工文本来源：{manual_record["source_filename"]}'
        f'；更新时间：{manual_record["updated_at"]}'
    )