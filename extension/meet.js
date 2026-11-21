// Runs on the Google Meet page.
// Watches the small "speaking" animation inside each participant's tile and tells
// the Meeting Agent page who is speaking, and how long ago that person started.
// It reads only names and the animation. Nothing is sent anywhere except to the extension itself.
// DEBUG version: prints which id belongs to which name, which ids are active,
// and whether sending to the extension worked.

(() => {
  const WINDOW_MS = 1000;                        // look at the animation of the last second
  const MIN_SCORE = 2;                           // fewer changes than this = not speaking
  const MARGIN = 1;                              // another tile must beat the current one by this much
  const GAP_MS = 1200;                           // a pause this long starts a new "burst" of speech
  const HEARTBEAT_MS = 2000;                     // while someone keeps speaking, say so again this often
  const STILL_ACTIVE_MS = 400;                   // "still speaking" = the animation changed this recently
  const INDICATOR = ".DYfzY, .IisKdb, .qg7mD";   // class names of the speaking animation

  const ICON_WORDS = new Set([
    "devices", "keep", "mic", "mic_off", "videocam", "videocam_off", "pip",
    "more_vert", "push_pin", "visual_effects", "close", "check", "frame_person"
  ]);

  // text of buttons that Meet shows inside a tile when the mouse is over it
  const UI_LABEL = /effects|show in a tile|more options|pin |unpin|mute|minimi[sz]e|remove|full ?screen|spotlight|present|reframe|backgrounds/i;

  function looksLikeIcon(s) {
    if (ICON_WORDS.has(s)) return true;
    return /^[a-z0-9]+(_[a-z0-9]+)+$/.test(s);
  }

  const events = {};                             // participant id -> times of recent animation changes
  const lastEvent = {};                          // participant id -> time of the last change
  const burstStart = {};                         // participant id -> when the current speech began
  const counts = {};                             // participant id -> changes since the last debug line
  const nameCache = {};                          // participant id -> clean name
  let current = null;
  let announcedKey = "";                         // "id:burst" of the last speaker message sent
  let lastAnnounce = 0;                          // when the last speaker message was sent
  let sendFailed = false;

  function send(msg) {
    try {
      const p = chrome.runtime.sendMessage(msg);
      if (p && p.catch) {
        p.catch(e => console.log("[Meeting Agent] send failed (async):", e && e.message));
      }
      if (sendFailed) {
        sendFailed = false;
        console.log("[Meeting Agent] sending works again");
      }
    } catch (e) {
      if (!sendFailed) {
        sendFailed = true;
        console.log("[Meeting Agent] send failed:", e && e.message,
          "-> reload the extension, then close this Meet tab and open it again");
      }
    }
  }

  // Finds the display name in the text lines of a tile. Returns null if it is not sure.
  function nameFromLines(lines) {
    for (let i = 0; i + 1 < lines.length; i++) {
      if (lines[i] === lines[i + 1] && !looksLikeIcon(lines[i]) && !UI_LABEL.test(lines[i])) {
        return { name: lines[i], sure: true };
      }
    }
    for (const l of lines) {
      const m = l.match(/^More options for (.+)$/);
      if (m) return { name: m[1].trim(), sure: true };
    }
    const other = lines.find(s => !looksLikeIcon(s) && !UI_LABEL.test(s));
    return other ? { name: other, sure: false } : null;
  }

  // Returns the name, or null while it is not known yet (so the caller can wait).
  function nameOf(id) {
    if (nameCache[id]) return nameCache[id];

    const item = document.querySelector('[role="listitem"][data-participant-id="' + id + '"]');
    const label = item && item.getAttribute("aria-label");
    if (label && !looksLikeIcon(label) && !UI_LABEL.test(label)) {
      nameCache[id] = label.trim();
      return nameCache[id];
    }

    const tile = document.querySelector('[data-tile-media-id="' + id + '"]');
    if (tile) {
      const lines = (tile.innerText || "").split("\n").map(s => s.trim()).filter(Boolean);
      const found = nameFromLines(lines);
      if (found) {
        if (found.sure) nameCache[id] = found.name;   // only save names we are sure about
        return found.name;
      }
    }
    return null;
  }

  function shortId(id) {
    return id.split("/").pop();
  }

  // number of animation changes of this participant in the last second
  function score(id, now) {
    const list = events[id];
    if (!list) return 0;
    while (list.length && now - list[0] > WINDOW_MS) list.shift();
    return list.length;
  }

  // 1. note every change of the speaking animation, and when each burst of speech began
  const observer = new MutationObserver(mutations => {
    const now = Date.now();
    for (const m of mutations) {
      if (m.attributeName !== "class") continue;
      const el = m.target;
      if (!el.matches || !el.matches(INDICATOR)) continue;
      const holder = el.closest("[data-participant-id]");
      if (holder) {
        const id = holder.getAttribute("data-participant-id");
        if (!lastEvent[id] || now - lastEvent[id] > GAP_MS) burstStart[id] = now;
        lastEvent[id] = now;
        (events[id] = events[id] || []).push(now);
        counts[id] = (counts[id] || 0) + 1;
      }
    }
  });
  observer.observe(document.body, { attributes: true, attributeFilter: ["class"], subtree: true });

  // 2. decide who is speaking (10 times a second):
  //    the tile with the most animation in the last second, if it beats the current one
  setInterval(() => {
    const now = Date.now();
    let best = null, bestScore = 0;
    for (const id of Object.keys(events)) {
      const s = score(id, now);
      if (s > bestScore) { best = id; bestScore = s; }
    }
    if (!best || bestScore < MIN_SCORE) return;         // silence: keep the last speaker
    const key = best + ":" + burstStart[best];
    const repeat = key === announcedKey;                // true: the same burst, reported again
    if (repeat) {                         // this burst of speech was already reported:
      // repeat it now and then while the person is still speaking, so that a page that started
      // later (or missed the first message) still learns who is speaking
      const stillSpeaking = now - (lastEvent[best] || 0) < STILL_ACTIVE_MS;
      if (!(stillSpeaking && now - lastAnnounce >= HEARTBEAT_MS)) return;
    }
    if (best !== current) {                             // a different person: must be clearly stronger
      const currentScore = current ? score(current, now) : 0;
      if (bestScore < currentScore + MARGIN) return;
    }
    const name = nameOf(best);
    if (!name) return;                                  // name not known yet: try again
    current = best;
    announcedKey = key;
    lastAnnounce = now;
    const ago = Math.max(1, Math.min(3000, now - (burstStart[best] || now)));
    console.log("[Meeting Agent] speaker:", name,
      "(id " + shortId(current) + ", score " + bestScore + ", began " + ago + " ms ago" +
      (repeat ? ", repeat" : "") + ")");
    send({ type: "speaker", name, ago: repeat ? -ago : ago });   // a negative number marks a repeat
  }, 100);

  // DEBUG: once a second, show which ids had animation changes
  setInterval(() => {
    const ids = Object.keys(counts);
    if (ids.length === 0) return;
    console.log("[Meeting Agent] activity:",
      ids.map(id => (nameOf(id) || "?") + " (id " + shortId(id) + ") = " + counts[id]).join(" | "));
    ids.forEach(id => delete counts[id]);
  }, 1000);

  // 3. tell the agent page who is in the meeting (sent every 3 s, so a page that was
  //    opened or reloaded later still receives it; logged only when it changes)
  let lastKey = "";
  setInterval(() => {
    const ids = [...new Set([...document.querySelectorAll("[data-tile-media-id]")]
      .map(e => e.getAttribute("data-participant-id") || e.getAttribute("data-tile-media-id")))];
    const named = ids.map(id => ({ id, name: nameOf(id) })).filter(x => x.name);
    const names = [...new Set(named.map(x => x.name))];
    if (names.length === 0) return;

    const key = named.map(x => x.id + "=" + x.name).join("|");
    if (key !== lastKey) {
      lastKey = key;
      console.log("[Meeting Agent] participants:", names.join(", "));
      ids.forEach(id => {
        const tile = document.querySelector('[data-tile-media-id="' + id + '"]');
        const raw = tile ? (tile.innerText || "").split("\n").map(s => s.trim()).filter(Boolean) : [];
        console.log("[Meeting Agent] map: id " + shortId(id) + " -> " + (nameOf(id) || "?"),
          "| tile text:", JSON.stringify(raw));
      });
    }
    send({ type: "participants", names });
  }, 3000);

  console.log("[Meeting Agent] meet.js loaded (version 7: repeats while someone speaks)");
})();