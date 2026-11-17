import { ScribeApp } from "./ScribeApp.js";

// Module scripts run after the document is parsed, so the DOM is ready here.
window.scribe = new ScribeApp(document);
