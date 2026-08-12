export function bindShowcaseInteractions({ citationText }) {
  const body = document.body;
  document.querySelectorAll("[data-menu-open]").forEach((button) => button.addEventListener("click", () => body.classList.add("menu-open")));
  document.querySelectorAll("[data-menu-close], .nav-link").forEach((button) => button.addEventListener("click", () => body.classList.remove("menu-open")));

  document.querySelector("[data-copy-citation]")?.addEventListener("click", async (event) => {
    try {
      await navigator.clipboard.writeText(citationText);
      event.currentTarget.textContent = "Copied";
      window.setTimeout(() => { event.currentTarget.textContent = "Copy citation"; }, 1600);
    } catch {
      event.currentTarget.textContent = "Copy unavailable";
    }
  });

  document.querySelectorAll("[data-selectable-card]").forEach((card) => {
    card.addEventListener("click", () => {
      const wasSelected = card.classList.contains("is-selected");
      const group = card.closest("[data-card-group]");
      if (group?.dataset.selectionMode !== "multiple") {
        group?.querySelectorAll("[data-selectable-card]").forEach((peer) => {
          peer.classList.remove("is-selected");
          peer.setAttribute("aria-pressed", "false");
        });
      }
      if (!wasSelected) {
        card.classList.add("is-selected");
        card.setAttribute("aria-pressed", "true");
      } else {
        card.classList.remove("is-selected");
        card.setAttribute("aria-pressed", "false");
      }
    });
  });

  document.querySelectorAll("[data-language-option]").forEach((option) => {
    option.addEventListener("click", () => {
      const orbit = option.closest(".language-card")?.querySelector("[data-language-orbit]");
      if (!orbit) return;
      const isSelected = option.getAttribute("aria-pressed") === "true";
      const value = orbit.querySelector("strong");
      const label = orbit.querySelector("span");
      orbit.classList.toggle("has-active-language", isSelected);
      orbit.style.setProperty("--active-start", `${option.dataset.start}%`);
      orbit.style.setProperty("--active-end", `${option.dataset.end}%`);
      orbit.style.setProperty("--active-color", option.dataset.color);
      value.textContent = isSelected ? option.dataset.short : orbit.dataset.defaultValue;
      label.textContent = isSelected ? option.dataset.label : orbit.dataset.defaultLabel;
    });
  });

  const links = [...document.querySelectorAll(".nav-link")];
  const targets = links.map((link) => document.querySelector(link.hash)).filter(Boolean);
  let framePending = false;
  const updateActiveLink = () => {
    const readingLine = window.scrollY + window.innerHeight * 0.22;
    const active = targets.reduce((current, target) => target.offsetTop <= readingLine ? target : current, targets[0]);
    links.forEach((link) => link.classList.toggle("active", link.hash === `#${active.id}`));
    framePending = false;
  };
  window.addEventListener("scroll", () => {
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(updateActiveLink);
  }, { passive: true });
  updateActiveLink();
}
