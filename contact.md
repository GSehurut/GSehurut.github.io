---
layout: default
title: Contact
description: Get in touch with Goabaone Sehurutshe about actuarial analytics, data engineering or opportunities in the London specialty insurance market.
permalink: /contact/
---

<div class="wrap page">
  <header class="page__header">
    <p class="eyebrow">Contact</p>
    <h1>Get in touch</h1>
    <p class="lead">
      Questions about actuarial analytics, data engineering in insurance, or an
      opportunity you'd like to discuss? I usually reply within a few days.
    </p>
  </header>

  <div class="contact">
    <ul class="contact__list plain-list">
      <li>
        <i class="fas fa-envelope" aria-hidden="true"></i>
        <a href="mailto:{{ site.author.email }}">{{ site.author.email }}</a>
      </li>
      <li>
        <i class="fab fa-linkedin-in" aria-hidden="true"></i>
        <a href="https://www.linkedin.com/in/{{ site.author.social.linkedin }}" rel="noopener">LinkedIn</a>
      </li>
      <li>
        <i class="fab fa-github" aria-hidden="true"></i>
        <a href="https://github.com/{{ site.author.social.github }}" rel="noopener">GitHub</a>
      </li>
      <li>
        <i class="fas fa-location-dot" aria-hidden="true"></i>
        {{ site.author.location }}
      </li>
    </ul>

    <form class="form card" action="https://formspree.io/f/xpzgwqzw" method="POST">
      <h2 class="card__title">Send a message</h2>
      <div class="form__field">
        <label for="name">Name</label>
        <input type="text" id="name" name="name" autocomplete="name" required />
      </div>
      <div class="form__field">
        <label for="email">Email</label>
        <input type="email" id="email" name="email" autocomplete="email" required />
      </div>
      <div class="form__field">
        <label for="subject">Topic</label>
        <select id="subject" name="subject" required>
          <option value="">Choose a topic</option>
          <option value="opportunity">Job or project opportunity</option>
          <option value="actuarial">Actuarial analytics</option>
          <option value="data-engineering">Data engineering</option>
          <option value="writing">Something I wrote</option>
          <option value="other">Something else</option>
        </select>
      </div>
      <div class="form__field">
        <label for="message">Message</label>
        <textarea id="message" name="message" rows="6" required></textarea>
      </div>
      <p class="muted form__note">All fields are required.</p>
      <button type="submit" class="button button--primary">
        <i class="fas fa-paper-plane" aria-hidden="true"></i> Send message
      </button>
    </form>
  </div>
</div>
