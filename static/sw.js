/*
 * fdash service worker: receives Web Push messages from the alerts worker and shows them as
 * system notifications, even when the dashboard tab is closed or the browser is in the background.
 *
 * Served by Streamlit (server.enableStaticServing) at <base>/app/static/sw.js; its scope is that folder,
 * which is all that is needed for push (push events are delivered to the registration, not to a page).
 */
const SCOPE = self.registration.scope;                 // https://host/<base>/app/static/
const APP_ROOT = new URL("../../", SCOPE).href;        // https://host/<base>/   (the dashboard)

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("push", (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch (e) {
    data = { body: event.data ? event.data.text() : "" };
  }
  const title = data.title || "fdash";
  const options = {
    body: data.body || "",
    icon: new URL(data.icon || "icon-192.png", SCOPE).href,
    badge: new URL(data.badge || "badge-72.png", SCOPE).href,
    tag: data.tag || "fdash-alert",        // same rule -> replaces the previous notification
    renotify: data.renotify !== false,     // ...but still vibrates / makes sound
    timestamp: data.timestamp || Date.now(),
    vibrate: data.vibrate || [120, 60, 120],
    actions: data.actions || [],
    data: { url: data.url || "/" },
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  if (event.action === "dismiss") return;
  event.waitUntil((async () => {
    const wins = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    for (const w of wins) {
      if (w.url.startsWith(APP_ROOT) && "focus" in w) return w.focus();
    }
    return self.clients.openWindow(APP_ROOT);
  })());
});

// If the browser rotates the subscription, the dashboard re-syncs it the next time it is opened
// (the Alerts section calls pushManager.getSubscription() on load and re-registers it).
self.addEventListener("pushsubscriptionchange", () => {});
