import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout } from "node:timers/promises";

const [url, output = "grafana-roll-rlvr.png", expectedPanels = "2"] = process.argv.slice(2);
if (!url) throw Error("Usage: node capture-dashboard.mjs <dashboard-url> [output.png] [panel-count]");
const profile = await mkdtemp(join(tmpdir(), "roll-grafana-chrome-"));
const child = spawn(process.env.CHROME || "google-chrome", [
  "--headless", "--no-proxy-server", "--disable-dev-shm-usage",
  "--user-data-dir=" + profile, "--window-size=1500,840",
  "--remote-debugging-port=0", "about:blank",
], { stdio: "ignore" });

async function until(task, limit = 30000) {
  const deadline = Date.now() + limit;
  let last;
  while (Date.now() < deadline) {
    try { const result = await task(); if (result) return result; }
    catch (error) { last = error; }
    await setTimeout(250);
  }
  throw Error("Timed out waiting for dashboard", { cause: last });
}
let ws;
try {
  const port = await until(async () => (await readFile(join(profile, "DevToolsActivePort"), "utf8")).split("\n")[0]);
  const tab = await until(async () => {
    const tabs = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    return tabs.find(t => t.type === "page");
  });
  ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, { once: true });
    ws.addEventListener("error", reject, { once: true });
  });
  let seq = 0;
  const pending = new Map();
  const queries = new Map();
  ws.addEventListener("message", ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.id) {
      const waiter = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) waiter?.reject(Error(msg.error.message));
      else waiter?.resolve(msg.result);
    } else if (msg.method === "Network.responseReceived" &&
               /\/api\/(ds\/query|.*\/api\/search)/.test(msg.params.response.url)) {
      queries.set(msg.params.requestId, msg.params.response.status);
    }
  });
  const cmd = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const evalBrowser = async expression =>
    (await cmd("Runtime.evaluate", {expression, returnByValue: true})).result.value;
  await cmd("Network.enable");
  await cmd("Emulation.setDeviceMetricsOverride", {
    width: 1500, height: 840, deviceScaleFactor: 1, mobile: false,
  });
  await cmd("Page.navigate", {url});
  try {
    await until(async () => (await evalBrowser(
      'document.querySelectorAll(".uplot canvas").length'
    )) >= Number(expectedPanels), 55000);
  } catch (error) {
    console.error(await evalBrowser("document.body.innerText"));
    console.error("Datasource statuses:", [...queries.values()]);
    throw error;
  }
  const {data} = await cmd("Page.captureScreenshot", {format: "png", captureBeyondViewport: false});
  await writeFile(output, Buffer.from(data, "base64"));
  console.log("Captured", output, "with datasource statuses", [...queries.values()]);
} finally {
  ws?.close();
  child.kill();
  await setTimeout(700);
  await rm(profile, {recursive:true, force:true});
}
