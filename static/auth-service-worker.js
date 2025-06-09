// Service Worker for IntegriPic
// This file prevents 404 errors when browsers request service workers

self.addEventListener('install', function(event) {
    console.log('Service Worker installed');
});

self.addEventListener('activate', function(event) {
    console.log('Service Worker activated');
});

self.addEventListener('fetch', function(event) {
    // Let the browser handle all requests normally
    return;
});
