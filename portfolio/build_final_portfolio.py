"""Build the four-page portfolio from verified project facts and supplied screenshots."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ModelPicker-Portfolio.pdf'
URL = 'https://claude.ai/share/57d33b70-7408-42b1-aa20-c7fb3cb21f9d'
for name, file in [('Body', 'DejaVuSans.ttf'), ('Bold', 'DejaVuSans-Bold.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name, '/usr/share/fonts/truetype/dejavu/' + file))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')
W, H = 595.276, 841.89
INK, MUTED, TEAL, PALE, LINE = [colors.HexColor(x) for x in ['#142539', '#536477', '#087F80', '#EAF5F3', '#DCE5EB']]
c = canvas.Canvas(str(OUT), pagesize=(W, H))
c.setTitle('ModelPicker MCP | Project Portfolio')
c.setAuthor('ModelPicker MCP')
c.setSubject('Verified capabilities, connection guide, and three pages of live Claude Desktop screenshots')


def text(s, x, y, w=507, size=9, bold=False, color=INK, leading=None, font=None):
    p = Paragraph(s, ParagraphStyle('body', fontName=font or ('Bold' if bold else 'Body'),
                                  fontSize=size, leading=leading or size * 1.45, textColor=color))
    _, h = p.wrap(w, H)
    assert y < 44 or y - h > 46, f'Content crosses footer: {s[:70]}'
    p.drawOn(c, x, y-h)
    return y-h


def label(s, y, x=44):
    return text(s.upper(), x, y, size=8, bold=True, color=TEAL)


def page(n, section):
    left, right = colors.HexColor('#121A29'), colors.HexColor('#363078')
    for i in range(300):
        t = i / 299
        c.setFillColor(colors.Color(left.red*(1-t)+right.red*t, left.green*(1-t)+right.green*t, left.blue*(1-t)+right.blue*t))
        c.rect(i*W/300, H-45, W/300+0.2, 45, fill=1, stroke=0)
    c.setFillColor(colors.HexColor('#5545EE'))
    c.circle(47, H-23, 7, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.circle(47, H-23, 2.4, fill=1, stroke=0)
    text('ModelPicker MCP', 62, H-16, w=300, size=10, bold=True, color=colors.white)
    c.setFont('Bold', 6.5)
    c.setFillColor(colors.HexColor('#B9BEF5'))
    c.drawRightString(W-44, H-25, 'PROJECT OVERVIEW' if n == 1 else 'LIVE SESSION')
    c.setStrokeColor(LINE)
    c.line(44, 43, W-44, 43)
    text('ModelPicker MCP', 44, 33, size=7, color=MUTED)
    c.setFillColor(MUTED)
    c.setFont('Body', 7)
    c.drawRightString(W-44, 23, f'{n} / 4')


def image(path, y, w=507, x=44):
    reader = ImageReader(str(path))
    iw, ih = reader.getSize()
    h = ih*w/iw
    assert y-h > 46
    c.drawImage(reader, x, y-h, w, h, mask='auto')
    return y-h


def note(s, y):
    p = Paragraph(s, ParagraphStyle('note', fontName='Body', fontSize=8, leading=12, textColor=MUTED))
    _, h = p.wrap(483, H)
    assert y-h-20 > 46
    c.setFillColor(PALE)
    c.roundRect(44, y-h-20, 507, h+20, 7, fill=1, stroke=0)
    p.drawOn(c, 56, y-h-10)
    return y-h-20


page(1, 'Project capabilities & connection guide')
text('From chat to measured results.', 44, 780, size=25, bold=True)
text('Discover. Evaluate. Choose with evidence.', 44, 739, size=12, color=TEAL)
text('ModelPicker turns a Claude conversation into a reproducible CPU benchmark of text classifiers. Discover Hugging Face models, import labeled data, compare quality and speed, and export results. Supports single-label sentiment, spam and ticket classification.', 44, 712, size=9, color=MUTED, leading=14)
c.setFillColor(colors.HexColor('#F0EFFB'))
c.roundRect(44, 609, 507, 44, 8, fill=1, stroke=0)
for i, (value, caption) in enumerate([('23', 'MCP tools'), ('5', 'models per plan'), ('1,000', 'rows per model'), ('CPU', 'local inference')]):
    x = 58+i*126
    text(value, x, 645, w=112, size=16, bold=True, color=colors.HexColor('#4539AA'))
    text(caption, x, 623, w=112, size=7, color=MUTED)
for i, (title, body) in enumerate([
    ('Discover & import', 'Search models and datasets; inspect labels and revisions. Import CSV, JSONL, JSON, pasted rows, trusted URLs or Hub datasets.'),
    ('Plan & evaluate', 'Seeded sampling, pinned revisions and identical rows. Background jobs with progress, cancellation and resource limits.'),
    ('Measure & reuse', 'Accuracy, macro F1, per-label scores, latency, throughput and memory. Save runs; export reports and Python integration code.'),
]):
    x=44+i*173
    c.setFillColor(colors.HexColor('#F5F7FA'))
    c.roundRect(x, 492, 161, 103, 7, fill=1, stroke=0)
    text(title, x+11, 583, w=139, size=9, bold=True)
    text(body, x+11, 562, w=139, size=7.5, color=MUTED, leading=11)
text('Python · MCP / FastMCP · PyTorch · Transformers · Hugging Face · SQLite', 44, 481, size=7, color=MUTED)
label('01 / Local connection', 459)
text('1. In the project environment, run <font face="Mono">python -m modelpicker setup</font>.<br/>2. Merge <font face="Mono">~/.modelpicker/desktop-config.json</font> → <font face="Mono">mcpServers.modelpicker</font><br/>into Claude’s config: Desktop app → Local MCP servers → Edit config.<br/>3. Fully quit and reopen Claude Desktop; enable ModelPicker in the chat tools menu.<br/><b>stdio · No ModelPicker OAuth · Uses this computer’s CPU</b>', 44, 441, size=8, leading=12)
label('02 / Remote connection', 366)
text('1. Run setup, then create a public HTTPS tunnel:<br/><font face="Mono">cloudflared tunnel --url http://127.0.0.1:7860</font><br/>2. Use the tunnel’s HTTPS origin to start the server (one command):<br/><font face="Mono">python -m modelpicker serve --transport streamable-http<br/>  --public-url https://YOUR-TUNNEL-HOST</font><br/>3. Claude Settings → Connectors → Add custom connector:<br/><font face="Mono">https://YOUR-TUNNEL-HOST/mcp</font>. Authorize in the browser using<br/><font face="Mono">~/.modelpicker/owner-password.txt</font>. Keep the server and tunnel running.', 44, 348, size=8, leading=12)
text('Uses the server host’s CPU. Quick-tunnel URLs change on restart; reconnect after server restarts. One trusted owner per installation. Use one HTTP service for multiple clients; avoid concurrent servers sharing storage.', 44, 242, size=7, color=MUTED, leading=10)
label('Connected in Claude Desktop', 204)
image(ROOT/'claude-local-server.png', 186, 265, 44)
image(ROOT/'claude-tools-enabled.png', 186, 226, 325)
text('Text and tool results sent to Claude reach the provider. No paid inference API is required; downloads and hardware costs still apply.', 44, 67, size=6.5, color=MUTED, leading=9)
c.showPage()


def live(n, title, prompt, tools):
    page(n, 'Live session')
    y = text(title, 44, 780, size=22, bold=True)
    y = text(prompt, 44, y-8, size=10, bold=True, leading=14)
    y = text('Implemented workflow tools: '+tools, 44, y-9, size=7, color=TEAL, leading=11)
    y = text(f'<link href="{URL}" color="#087F80"><u>View conversation</u></link>', 44, y-6, size=7, color=MUTED)
    return y-14


y = live(2, '01 / Inspect capabilities', '“Inspect my modelpicker MCP and show its capabilities”', 'get_capabilities')
y = image(ROOT/'session-assets/session-002.png', y)
y = image(ROOT/'session-assets/session-003.png', y-10)
note('The session reports the local CPU environment, package versions and resource limits. These are historical observations from this session. Every tool name visible in the screenshot exists in the project.', y-16)
c.showPage()

y = live(3, '02 / Discover models & datasets', '“Search top sentiment classify models for me and also search for a dataset on huggingface”', 'search_models · search_datasets')
y = image(ROOT/'session-assets/session-004.png', y, 490, 52.5)
y = image(ROOT/'session-assets/session-005.png', y-10, 490, 52.5)
note('These are discovery results, not an evaluated ranking. Download counts, licenses and model metadata reflect the recorded session; compatibility requires inspection before evaluation.', y-12)
c.showPage()

y = live(4, '03 / Evaluate & compare', '“Take the first dataset and evaluate the top 3 models, give me a comparison result”', 'inspect_huggingface_dataset · import_huggingface_dataset · inspect_model · create_evaluation_plan · start_evaluation · get_evaluation')
for i in [6, 7, 8]:
    y = image(ROOT/f'session-assets/session-{i:03d}.png', y, 490, 52.5)
    y -= 8
note('Verified against the saved evaluation: 200 English test rows (188 positive, 12 negative), 600 predictions, no execution errors. Accuracy: RoBERTa 96%, BERT 93%, DistilBERT 92%. Imbalanced, exploratory sample. Inference latency/throughput exclude download, loading and warmup; memory and weight figures use MiB.', y-4)
c.save()
print(OUT)
