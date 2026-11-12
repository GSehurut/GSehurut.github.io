# gsehurut.github.io

Personal site of **Goabaone Sehurutshe**, Actuarial Analytics Engineer at The Fidelis Partnership, London. Built with Jekyll and hosted on GitHub Pages: <https://gsehurut.github.io>

## Updating content

| What | Where |
| --- | --- |
| Role, summary, experience, education, qualifications, skills | `_data/cv.yml` (feeds the home and Experience pages) |
| Articles | `_posts/YYYY-MM-DD-slug.md` with `layout: post`, `title`, `description`, `categories`, `tags` |
| Projects | `projects/slug.md` with `layout: project`, `title`, `date`, `description`, `categories`, `tags` (listed automatically on `/projects/`) |
| About / Contact | `about/index.html`, `contact.md` |
| Name, email, social links | `_config.yml` → `author` |
| Styles | `css/main.css` — design tokens at the top; dark mode via `prefers-color-scheme` |

The sitemap (`jekyll-sitemap`), RSS feed (`jekyll-feed`) and SEO/Open Graph tags (`jekyll-seo-tag`) are generated at build time.

## Running locally

```sh
gem install jekyll jekyll-feed jekyll-seo-tag jekyll-sitemap webrick
jekyll serve --livereload
# open http://localhost:4000
```

## Structure

```
_config.yml        site settings and author details
_data/cv.yml       CV content
_includes/         shared partials (post card)
_layouts/          default, post and project templates
_posts/            articles
about/ blog/ experience/ projects/   section pages
css/main.css       styles
pictures/          images
```
