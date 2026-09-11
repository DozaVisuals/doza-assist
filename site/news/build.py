#!/usr/bin/env python3
"""Build the doza.ai newsroom from docs/articles/*.md.

    python3 site/news/build.py            # writes site/news/index.html + one folder per post

Each markdown file needs YAML-ish front matter (title, slug, date, author,
description, category). Everything after the front matter is the body.
"""
import html, re, sys, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'docs' / 'articles'
OUT = Path(__file__).resolve().parent
CSS = (OUT / 'news.css').read_text()

SITE = 'https://doza.ai'
LI = re.compile(r'^(- |\d+\. )')

def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<!\*)\*(?!\*)(.+?)\*', r'<em>\1</em>', s)
    s = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', s)
    return s

def md_to_html(body):
    out = []
    for para in re.split(r'\n\s*\n', body.strip()):
        lines = para.split('\n')
        if para.startswith('# '):
            continue  # title comes from front matter
        if para.startswith('## '):
            out.append('<h2>' + inline(para[3:]) + '</h2>')
        elif para.startswith('### '):
            out.append('<h3>' + inline(para[4:]) + '</h3>')
        elif para.strip() == '---':
            out.append('<hr>')
        elif all(LI.match(l) for l in lines):
            tag = 'ol' if lines[0][0].isdigit() else 'ul'
            out.append('<%s>%s</%s>' % (tag, ''.join('<li>' + inline(LI.sub('', l)) + '</li>' for l in lines), tag))
        elif para.startswith('> '):
            out.append('<blockquote><p>' + inline(' '.join(l.lstrip('> ') for l in lines)) + '</p></blockquote>')
        else:
            out.append('<p>' + inline(' '.join(lines)) + '</p>')
    return '\n'.join(out)

def parse(path):
    text = path.read_text()
    _, fm, body = text.split('---', 2)
    meta = {}
    for line in fm.strip().splitlines():
        k, _, v = line.partition(':')
        meta[k.strip()] = v.strip().strip('"')
    meta['body'] = body
    meta['date_obj'] = datetime.date.fromisoformat(meta['date'])
    meta['date_h'] = meta['date_obj'].strftime('%b %-d, %Y')
    meta.setdefault('category', 'Perspectives')
    meta.setdefault('author', 'Doza Visuals')
    return meta

def shell(title, description, canonical, body, is_article=False, meta=None):
    og = ''
    if is_article and meta:
        og = f'''
<meta property="article:published_time" content="{meta['date']}">
<meta property="article:author" content="{html.escape(meta['author'])}">'''
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description, quote=True)}">
<link rel="canonical" href="{canonical}">
<meta property="og:type" content="{'article' if is_article else 'website'}">
<meta property="og:title" content="{html.escape(title, quote=True)}">
<meta property="og:description" content="{html.escape(description, quote=True)}">
<meta property="og:url" content="{canonical}">
<meta property="og:site_name" content="Doza">
<meta name="twitter:card" content="summary_large_image">{og}
<style>
{CSS}
</style>
</head>
<body>
<header class="nav">
  <div class="nav-inner">
    <a class="wordmark" href="{SITE}/">Doza Assist</a>
    <nav>
      <a href="{SITE}/">Doza Assist</a>
      <a href="{SITE}/news/" aria-current="{'page' if not is_article else 'false'}">News</a>
      <a href="https://github.com/DozaVisuals/doza-assist">GitHub</a>
      <a href="https://discord.gg/TTM3hWXM8">Discord</a>
    </nav>
  </div>
</header>
{body}
<footer class="foot">
  <div class="foot-inner">
    <div>
      <a class="wordmark small" href="{SITE}/">Doza Assist</a>
      <p class="foot-tag">Local AI for editors who find the story in the footage.</p>
    </div>
    <div class="foot-cols">
      <div><h4>Product</h4><a href="{SITE}/">Doza Assist</a><a href="https://github.com/DozaVisuals/doza-assist">Doza Assist Core</a></div>
      <div><h4>Company</h4><a href="{SITE}/news/">News</a><a href="mailto:chris@dozavisuals.com">Contact</a></div>
    </div>
  </div>
  <div class="foot-inner legal"><span>© {datetime.date.today().year} Doza Visuals</span></div>
</footer>
</body>
</html>
'''

def article_page(a, others):
    related = ''
    if others:
        cards = ''.join(f'''
      <a class="card" href="{SITE}/news/{o['slug']}/">
        <span class="eyebrow">{html.escape(o['category'])} · {o['date_h']}</span>
        <h3>{html.escape(o['title'])}</h3>
      </a>''' for o in others[:3])
        related = f'<section class="related"><div class="related-inner"><h2 class="section-label">More from Doza</h2><div class="cards">{cards}</div></div></section>'
    else:
        related = f'''<section class="related"><div class="related-inner"><h2 class="section-label">Keep reading</h2><div class="cards">
      <a class="card" href="https://github.com/DozaVisuals/doza-assist"><span class="eyebrow">Open source</span><h3>Doza Assist Core on GitHub</h3><p>Read the transcription pipeline yourself. MIT licensed.</p></a>
      <a class="card" href="{SITE}/"><span class="eyebrow">Product</span><h3>Doza Assist for Mac</h3><p>Multi-interview and press-ready workflows on the same local foundation.</p></a>
      <a class="card" href="https://discord.gg/TTM3hWXM8"><span class="eyebrow">Community</span><h3>Join the Discord</h3><p>Editors comparing notes on local AI and story-first cutting.</p></a>
    </div></div></section>'''
    url = f"{SITE}/news/{a['slug']}/"
    body = f'''
<main class="article">
  <header class="article-head">
    <p class="eyebrow"><a href="{SITE}/news/">{html.escape(a['category'])}</a><span class="dot">·</span><time datetime="{a['date']}">{a['date_h']}</time></p>
    <h1>{html.escape(a['title'])}</h1>
    <p class="dek">{html.escape(a['description'])}</p>
  </header>
  <div class="prose">
{md_to_html(a['body'])}
  </div>
  <aside class="share">
    <span>Share</span>
    <a href="https://x.com/intent/tweet?url={url}&text={html.escape(a['title'], quote=True)}">X</a>
    <a href="https://www.linkedin.com/sharing/share-offsite/?url={url}">LinkedIn</a>
    <a href="mailto:?subject={html.escape(a['title'], quote=True)}&body={url}">Email</a>
  </aside>
</main>
{related}'''
    return shell(a['title'] + ' \\ Doza', a['description'], url, body, True, a)

def index_page(arts):
    lead, rest = arts[0], arts[1:]
    featured = f'''
  <a class="featured" href="{SITE}/news/{lead['slug']}/">
    <span class="eyebrow">{html.escape(lead['category'])} · {lead['date_h']}</span>
    <h2>{html.escape(lead['title'])}</h2>
    <p>{html.escape(lead['description'])}</p>
    <span class="readmore">Read the article</span>
  </a>'''
    rows = ''.join(f'''
    <a class="row" href="{SITE}/news/{a['slug']}/">
      <span class="eyebrow">{html.escape(a['category'])}</span>
      <h3>{html.escape(a['title'])}</h3>
      <time datetime="{a['date']}">{a['date_h']}</time>
    </a>''' for a in rest)
    listing = f'<div class="rows">{rows}</div>' if rest else ''
    body = f'''
<main class="news">
  <header class="news-head"><h1>News</h1><p>Announcements, perspectives, and notes from building local AI for editors.</p></header>
  {featured}
  {listing}
</main>'''
    return shell('News \\ Doza', 'Announcements and perspectives from Doza, makers of Doza Assist.', f'{SITE}/news/', body)

def main():
    arts = sorted((parse(p) for p in SRC.glob('*.md') if 'slug:' in p.read_text()), key=lambda a: a['date_obj'], reverse=True)
    if not arts:
        sys.exit('no articles found in docs/articles')
    for a in arts:
        d = OUT / a['slug']; d.mkdir(exist_ok=True)
        (d / 'index.html').write_text(article_page(a, [o for o in arts if o is not a]))
    (OUT / 'index.html').write_text(index_page(arts))
    print(f'built {len(arts)} article(s) into {OUT}')

if __name__ == '__main__':
    main()
