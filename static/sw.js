self.addEventListener("push", event => {
  let data = {};
  try { data = event.data ? event.data.json() : {}; } catch (_) {}

  const title = data.title || "Security Alert";
  const options = {
    body: data.body || "A security event was detected.",
    tag: data.incident_id ? `incident-${data.incident_id}` : "surveillance-alert",
    renotify: true,
    requireInteraction: data.requireInteraction === true,
    vibrate: data.vibrate || [500, 200, 500, 200, 800],
    data: { url: data.url || "/owner", incident_id: data.incident_id || null }
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", event => {
  event.notification.close();
  const url = event.notification.data?.url || "/owner";
  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true }).then(list => {
      for (const client of list) {
        if ("focus" in client) {
          client.focus();
          if ("navigate" in client) client.navigate(url);
          return;
        }
      }
      return clients.openWindow(url);
    })
  );
});
