(() => {
  const friction = 0.95;
  const moveFactor = 0.25;
  const stepFactor = 0.22;
  const minVelocity = 0.03;
  let target = window.scrollY;
  let velocity = 0;
  let isAnimating = false;
  let lastWheelTime = 0;
  let isUserDragging = false;

  function animateScroll() {
    if (!isAnimating || isUserDragging) return;
    velocity *= friction;
    target += velocity;
    const maxScroll = document.body.scrollHeight - window.innerHeight;
    target = Math.max(0, Math.min(target, maxScroll));
    const currentY = window.scrollY;
    const diff = target - currentY;
    const move = diff * moveFactor;
    window.scrollTo(0, currentY + move);
    if (Math.abs(velocity) > minVelocity || Math.abs(diff) > 0.5) {
      requestAnimationFrame(animateScroll);
    } else {
      isAnimating = false;
    }
  }

  function handleScroll(event) {
    if (isUserDragging || event.ctrlKey) return;
    event.preventDefault();
    const now = performance.now();
    const delta = event.deltaY || -event.wheelDelta || event.detail;
    if (now - lastWheelTime > 100) velocity = 0;
    lastWheelTime = now;
    velocity += delta * stepFactor;
    target = window.scrollY + velocity;
    if (!isAnimating) {
      isAnimating = true;
      requestAnimationFrame(animateScroll);
    }
  }

  function handleKeyScroll(event) {
    if (isUserDragging) return;
    const step = window.innerHeight * 0.9;
    if (["ArrowDown", "PageDown", " "].includes(event.key)) {
      event.preventDefault();
      velocity += step * 0.5;
      target += step;
      if (!isAnimating) {
        isAnimating = true;
        requestAnimationFrame(animateScroll);
      }
    } else if (["ArrowUp", "PageUp"].includes(event.key)) {
      event.preventDefault();
      velocity -= step * 0.5;
      target -= step;
      if (!isAnimating) {
        isAnimating = true;
        requestAnimationFrame(animateScroll);
      }
    }
  }

  window.addEventListener("mousedown", () => {
    isUserDragging = true;
    isAnimating = false;
  });

  window.addEventListener("mouseup", () => {
    isUserDragging = false;
  });

  window.addEventListener("scroll", () => {
    if (isUserDragging) {
      target = window.scrollY;
    }
  });

  window.addEventListener("wheel", handleScroll, { passive: false });
  window.addEventListener("keydown", handleKeyScroll);
})();
