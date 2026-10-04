// Bundles src/main.tsx -> dist/app.js and copies static files. `--watch` rebuilds on change;
// `--tests` bundles tests/*.test.ts for Node's built-in test runner.
import * as esbuild from "esbuild";
import { cpSync, mkdirSync, readdirSync, rmSync } from "node:fs";

const args = new Set(process.argv.slice(2));

if (args.has("--tests")) {
  rmSync("tests-dist", { recursive: true, force: true });
  const entries = readdirSync("tests").filter((f) => f.endsWith(".test.ts")).map((f) => `tests/${f}`);
  await esbuild.build({ entryPoints: entries, outdir: "tests-dist", bundle: true, platform: "node", format: "esm", outExtension: { ".js": ".mjs" } });
  process.exit(0);
}

rmSync("dist", { recursive: true, force: true });
mkdirSync("dist", { recursive: true });
cpSync("index.html", "dist/index.html");
cpSync("src/styles.css", "dist/styles.css");

const options = {
  entryPoints: ["src/main.tsx"],
  outfile: "dist/app.js",
  bundle: true,
  minify: !args.has("--watch"),
  sourcemap: args.has("--watch"),
  target: ["es2022"],
  define: { "process.env.NODE_ENV": JSON.stringify(args.has("--watch") ? "development" : "production") },
  logLevel: "info",
};

if (args.has("--watch")) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
} else {
  await esbuild.build(options);
}
