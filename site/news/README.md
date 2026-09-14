# doza.ai newsroom

Static news section for doza.ai, styled after the Anthropic newsroom in Doza Assist's own tokens.

- Write a post as markdown in `docs/articles/` with front matter (`title`, `slug`, `date`, `author`, `description`, `category`).
- Run `python3 site/news/build.py`. It writes `site/news/index.html` and `site/news/<slug>/index.html`.
- Upload the `site/news/` folder to the host so it serves at `https://doza.ai/news/`.
