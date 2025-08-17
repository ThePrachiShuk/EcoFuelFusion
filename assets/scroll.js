// Smooth scrolling enhancements and reveal-on-scroll effects for Dash
(function () {
	const onReady = (fn) => {
		if (document.readyState === "loading") {
			document.addEventListener("DOMContentLoaded", fn, { once: true });
		} else {
			fn();
		}
	};

	function getHeaderOffset() {
		const nav = document.querySelector(".navbar");
		// fallback in case navbar not present
		return (nav ? nav.getBoundingClientRect().height : 0) + 12; // a little extra spacing
	}

	function smoothScrollToHash(hash, replaceHistory = false) {
		if (!hash || hash === "#") return;
		const el =
			document.getElementById(hash.replace("#", "")) ||
			document.querySelector(hash);
		if (!el) return;

		const y =
			el.getBoundingClientRect().top + window.scrollY - getHeaderOffset();
		window.scrollTo({ top: y, behavior: "smooth" });

		if (replaceHistory && "history" in window) {
			history.replaceState(null, "", hash);
		}
	}

	function wireAnchorClicks(root) {
		const isSamePage = (a) =>
			a.hash &&
			a.origin === location.origin &&
			a.pathname === location.pathname;
		root.querySelectorAll('a[href^="#"]').forEach((a) => {
			a.addEventListener("click", (e) => {
				if (!isSamePage(a)) return; // external or different page
				e.preventDefault();
				smoothScrollToHash(a.hash, true);
			});
		});
	}

	function addRevealClasses(root) {
		// Mark common content blocks for reveal-on-scroll
		const candidates = root.querySelectorAll(
			[
				".card",
				".alert",
				".accordion-item",
				".modern-container",
				".section-title",
				".dash-spreadsheet-container",
			].join(",")
		);

		candidates.forEach((el, idx) => {
			if (!el.classList.contains("reveal-on-scroll")) {
				el.classList.add("reveal-on-scroll");
				// add a tiny stagger via CSS var
				el.style.setProperty(
					"--reveal-delay",
					`${Math.min((idx % 6) * 60, 240)}ms`
				);
			}
		});
	}

	function setupIntersectionObserver(root) {
		if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
			root
				.querySelectorAll(".reveal-on-scroll")
				.forEach((el) => el.classList.add("is-visible"));
			return;
		}

		const io = new IntersectionObserver(
			(entries) => {
				entries.forEach((entry) => {
					if (entry.isIntersecting) {
						entry.target.classList.add("is-visible");
						// Once visible, unobserve to avoid repeated work
						io.unobserve(entry.target);
					}
				});
			},
			{ threshold: 0.12 }
		);

		root.querySelectorAll(".reveal-on-scroll").forEach((el) => io.observe(el));
	}

	function bootstrap(root = document) {
		wireAnchorClicks(root);
		addRevealClasses(root);
		setupIntersectionObserver(root);

		// If page loads with a hash, adjust to offset
		if (location.hash) {
			// Delay to allow layout to settle
			setTimeout(() => smoothScrollToHash(location.hash, true), 50);
		}
	}

	// Initial boot
	onReady(() => {
		const root = document.getElementById("theme-root") || document;
		bootstrap(root);

		// Observe dynamic changes (Dash renders/updates) with debounce to reduce churn
		let refreshTimer = null;
		const mo = new MutationObserver((mutations) => {
			for (const m of mutations) {
				if (m.addedNodes && m.addedNodes.length) {
					if (refreshTimer) clearTimeout(refreshTimer);
					refreshTimer = setTimeout(() => {
						bootstrap(root);
						refreshTimer = null;
					}, 60); // small debounce window
					break;
				}
			}
		});
		mo.observe(root, { childList: true, subtree: true });
	});
})();
