const copyButton = document.querySelector("[data-copy-bibtex]");
const bibtexCode = document.querySelector("#bibtex-code");

if (copyButton && bibtexCode) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtexCode.textContent.trim());
      copyButton.classList.add("is-copied");
      copyButton.querySelector(".copy-label").textContent = "Copied";
      window.setTimeout(() => {
        copyButton.classList.remove("is-copied");
        copyButton.querySelector(".copy-label").textContent = "Copy";
      }, 1800);
    } catch (_error) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(bibtexCode);
      selection.removeAllRanges();
      selection.addRange(range);
    }
  });
}

const scrollTopButton = document.querySelector("[data-scroll-top]");

if (scrollTopButton) {
  const toggleScrollButton = () => {
    scrollTopButton.classList.toggle("is-visible", window.scrollY > 480);
  };

  window.addEventListener("scroll", toggleScrollButton, { passive: true });
  toggleScrollButton();
  scrollTopButton.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
}
