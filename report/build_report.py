"""
Builds report/PathSnap_Technical_Report.pdf from report/TECHNICAL_REPORT.md.

Pipeline: markdown --(pandoc)--> styled HTML --(headless Chrome via Playwright)--> PDF.
Every table and number lives in the markdown (generated from results/*.csv|json), and every
figure is an image file produced by the committed scripts, so the PDF contains nothing that
is not in the repository. If screenshots named report/figures/cmd_screenshot*.png exist
they are embedded under section 7.3 (real screenshots of Laptop 1's terminal; the caption lives in the markdown).

Requirements (only to rebuild the PDF; not needed to run the system): pandoc,
Google Chrome, and `pip install playwright`.
Usage: python report/build_report.py
"""
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "TECHNICAL_REPORT.md")
BUILD = os.path.join(HERE, "_build")
HTML = os.path.join(BUILD, "report.html")
PDF = os.path.join(HERE, "PathSnap_Technical_Report.pdf")
CHROME = os.environ.get("CHROME_PATH", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
body { font-family: -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1a202c; }
h1 { font-size: 21pt; color: #1a365d; margin: 0 0 2pt 0; }
h2 { font-size: 14.5pt; color: #1a365d; border-bottom: 2px solid #2b6cb0; padding-bottom: 3pt; margin-top: 20pt; page-break-before: always; }
#project-overview-and-summary-of-results { page-break-before: avoid; }
h1 + h2 { page-break-before: avoid; font-size: 12pt; color: #2b6cb0; border: none; margin-top: 0; }
h3 { font-size: 11.5pt; color: #2b6cb0; margin-top: 14pt; page-break-after: avoid; }
h4 { font-size: 10pt; color: #2d3748; margin-top: 10pt; page-break-after: avoid; }
p { margin: 5pt 0; text-align: justify; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.4pt; background: #edf2f7; padding: 0 2pt; border-radius: 2pt; }
pre { background: #1a202c; color: #e2e8f0; padding: 8pt; border-radius: 4pt; font-size: 7.4pt; line-height: 1.3; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; color: inherit; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0; font-size: 7.9pt; page-break-inside: avoid; }
th { background: #1a365d; color: #fff; padding: 3.5pt 4pt; text-align: center; font-weight: 600; }
td { border: 1px solid #cbd5e0; padding: 3pt 4pt; vertical-align: top; }
tr:nth-child(even) td { background: #f7fafc; }
img { display: block; margin: 8pt auto 2pt auto; max-width: 100%; }
p:has(> img) { page-break-after: avoid; page-break-inside: avoid; margin-bottom: 0; }
p:has(> img) + p > em:only-child { display: block; text-align: center; font-size: 8.3pt; color: #4a5568; margin-bottom: 8pt; }
hr { border: none; border-top: 1px solid #cbd5e0; margin: 10pt 0; }
ul, ol { margin: 4pt 0 4pt 16pt; padding: 0; }
li { margin: 2pt 0; }
"""


def main():
    if not shutil.which("pandoc"):
        sys.exit("pandoc not found (install pandoc to rebuild the PDF).")
    os.makedirs(BUILD, exist_ok=True)

    md = open(MD, encoding="utf-8").read()
    shots = sorted(glob.glob(os.path.join(HERE, "figures", "cmd_screenshot*.png")))
    block = "\n\n".join(f"![Laptop 1 terminal]({os.path.relpath(s, HERE)}){{width=100%}}" for s in shots)
    md = md.replace("<!--CMD_SCREENSHOTS-->", block)
    tmp_md = os.path.join(HERE, "_build", "report_src.md")
    open(tmp_md, "w", encoding="utf-8").write(md)
    open(os.path.join(BUILD, "style.css"), "w").write(CSS)

    # Run from report/ so the markdown's relative image paths (../results/..., figures/...) resolve.
    subprocess.run(
        ["pandoc", tmp_md, "-f", "markdown-implicit_figures+pipe_tables+link_attributes+raw_html",
         "-t", "html5", "--standalone", "--metadata", "title=PathSnap Technical Report", "--css", "style.css",
         "-V", "pagetitle=PathSnap Technical Report", "-o", HTML],
        check=True, cwd=BUILD,
    )
    # pandoc inserts a title header from metadata; the document already has its own title.
    html = open(HTML, encoding="utf-8").read()
    html = html.replace('<header id="title-block-header">', '<header id="title-block-header" style="display:none">')
    html = html.replace('src="../', 'src="../../').replace('src="figures/', 'src="../figures/')
    open(HTML, "w", encoding="utf-8").write(html)

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        page.goto("file://" + HTML)
        page.wait_for_load_state("networkidle")
        page.pdf(
            path=PDF, format="A4", print_background=True, display_header_footer=True,
            header_template="<div></div>",
            footer_template=("<div style='width:100%;font-size:7.5pt;color:#718096;text-align:center;"
                             "font-family:Helvetica,Arial,sans-serif'>PathSnap Technical Report &nbsp;|&nbsp; "
                             "page <span class='pageNumber'></span> of <span class='totalPages'></span></div>"),
            margin={"top": "16mm", "bottom": "18mm", "left": "14mm", "right": "14mm"},
        )
        browser.close()
    print("Wrote", PDF)


if __name__ == "__main__":
    main()
