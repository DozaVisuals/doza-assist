# Handoffs for Claude Code on the dev machine

Each block is a complete first message for a fresh Claude Code chat in the
doza-assist folder on the Mac that has GitHub authorized and Chrome logged into
Netlify. Run them in order; each one stands alone.

Context every chat needs: the newsroom lives in `site/news/` and is generated
from markdown in `docs/articles/` by `python3 site/news/build.py`. Posts dated
after the build date are held back. Pull request #43 on DozaVisuals/doza-assist
carries all of it on branch `claude/kind-volta-up6uhg`.

---

## 1. Connect doza.ai on Netlify to this repo

```
Pull request #43 on DozaVisuals/doza-assist adds a static newsroom under site/news
and a daily publish workflow at .github/workflows/publish-news.yml. Set up
continuous deployment so doza.ai is served from this repo.

1. git fetch, check out claude/kind-volta-up6uhg, pull.
2. In Chrome, open the doza.ai site in Netlify > Deploys > the current published
   deploy > Download. Unzip it into site/ so the homepage's index.html sits next to
   site/news/. Keep the branch's site/netlify.toml and site/_redirects; if the
   download has its own _redirects, fold its rules into site/_redirects. Do not
   overwrite anything in site/news/. Commit "site: bring doza.ai into the repo"
   and push.
3. Merge PR #43 into main.
4. In Netlify: Site configuration > Build & deploy > Continuous deployment > Link
   repository > GitHub > DozaVisuals/doza-assist. Production branch: main. Base
   directory: blank. Build command: blank. Publish directory: site. Save and wait
   for the first deploy.
5. Confirm https://doza.ai/ looks the same as before and https://doza.ai/news/
   and https://doza.ai/news/local-first-frontier-models-mcp/ both resolve.
Report what you found in the downloaded site (any redirects, forms, functions)
and anything that didn't carry over.
```

## 2. Wire the 8am publish job

```
The repo has .github/workflows/publish-news.yml, which runs daily at 12:00 UTC
(8am ET), builds posts in docs/articles dated today or earlier into site/news,
commits them to main, and POSTs to a Netlify build hook if the secret
NETLIFY_BUILD_HOOK exists. Finish wiring it.

1. In Chrome, Netlify > the doza.ai site > Site configuration > Build & deploy >
   Build hooks > Add build hook. Name "daily news", branch main. Copy the URL.
2. In GitHub, DozaVisuals/doza-assist > Settings > Secrets and variables > Actions
   > New repository secret. Name NETLIFY_BUILD_HOOK, value the hook URL.
3. GitHub > Actions > "Publish scheduled news posts" > Run workflow on main.
   Confirm it goes green. It's expected to commit nothing if today's posts are
   already built.
4. Check Netlify Deploys shows a deploy triggered by the hook.
Tell me the run URL and whether the hook fired.
```

## 3. Verify the week's schedule

```
Four posts are dated Tue 9/15 through Fri 9/18 in docs/articles and are held
back by site/news/build.py until their date. Verify the automation will publish
them on time.

1. Run `python3 site/news/build.py --as-of 2026-09-18` and confirm all six
   article folders appear in site/news, then run `python3 site/news/build.py`
   (no flag) and confirm the future ones are removed again. Don't commit the
   --as-of output.
2. Confirm the workflow cron is "0 12 * * *" and that the workflow file is on
   main (scheduled workflows only run from the default branch).
3. Open each future post's built page in a browser from the --as-of build and
   check: title, date, FAQ block renders, JSON-LD parses, the Tuesday and Friday
   posts link to each other. Fix anything broken in the markdown, rebuild, commit
   to main.
4. Check https://doza.ai/news/ shows exactly the posts due today.
```

## 4. Add an OpenGraph share image

```
The newsroom pages under site/news have no og:image, so link previews on X and
LinkedIn show text only. Add one.

1. Make a 1200x630 PNG per post: Doza Assist dark ground (#0a0a0c), the post
   title in SF Pro bold white (#e8e8ec), the category and date in the blue accent
   (#4a9eff) as a small uppercase eyebrow, "doza.ai" bottom-left in muted grey
   (#8e8e9a). Generate it with a script (Pillow or Playwright screenshot of an
   HTML template) so it runs for future posts too, and call it from
   site/news/build.py so each build writes site/news/<slug>/og.png.
2. Add <meta property="og:image"> and <meta name="twitter:image"> to the shell
   in build.py pointing at https://doza.ai/news/<slug>/og.png, plus a default
   image for the index page.
3. Rebuild, commit to main, push, and validate one URL with
   https://cards-dev.twitter.com/validator or opengraph.xyz after Netlify deploys.
```

## 5. Set up the next week's posts

```
Continue the doza.ai newsroom for next week. Read docs/articles/*.md to match
the voice and the front matter format (title, slug, date, author, description,
category), and end each post with a "## FAQ" section of 3-4 "### question"
blocks, which build.py turns into FAQPage schema.

Propose five topics with target search queries first, then on approval write
them dated Mon 9/21 through Fri 9/25 at docs/articles/YYYY-MM-DD-<slug>.md,
each 700-900 words, SEO title with the target phrase in the title, slug, and
first paragraph, question-shaped H2s, and internal links to existing posts.
Write a matching YYYY-MM-DD-social.md for each (LinkedIn/X long, X short,
hashtags). Ground every product claim in the codebase or README; don't invent
features. Commit to main and push; the 8am workflow publishes each on its date.
```

## 6. Submit the pages to search engines

```
The doza.ai newsroom is live at https://doza.ai/news/. Get it indexed.

1. Add a sitemap: extend site/news/build.py to write site/news/sitemap.xml
   listing the index and every published post with lastmod from the date, and
   add "Sitemap: https://doza.ai/news/sitemap.xml" to site/robots.txt (create it
   if the site has none, allowing everything). Rebuild, commit to main, push.
2. In Chrome, open Google Search Console for doza.ai (add the property with the
   DNS or HTML-file method if it isn't there yet), submit the sitemap, and request
   indexing for /news/ and each live post URL.
3. Do the same in Bing Webmaster Tools (it can import from Search Console).
Report which URLs were submitted and any errors Search Console shows.
```
