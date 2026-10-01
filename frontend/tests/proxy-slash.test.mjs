import test from "node:test";
import assert from "node:assert/strict";
import nextConfig from "../next.config.mjs";

test("Next.js configuration disables trailing slash canonicalization", async () => {
  assert.equal(
    nextConfig.skipTrailingSlashRedirect,
    true,
    "skipTrailingSlashRedirect must be true to prevent 308 redirects on trailing slashes"
  );

  const rewrites = await nextConfig.rewrites();
  assert.ok(Array.isArray(rewrites), "rewrites() should return an array");

  const apiRewrite = rewrites.find((r) => r.source === "/api/:path*");
  assert.ok(apiRewrite, "Must define a rewrite rule for /api/:path*");
  assert.ok(
    apiRewrite.destination.endsWith("/api/:path*/"),
    "Rewrite destination must preserve trailing slash to satisfy Django routing"
  );
});

test("Next.js server does not redirect /api/ routes with 308", async (t) => {
  try {
    const healthRes = await fetch("http://localhost:3000/api/health/", {
      redirect: "manual",
    });
    assert.notEqual(
      healthRes.status,
      308,
      "/api/health/ should not return 308 Permanent Redirect"
    );
    assert.equal(
      healthRes.headers.get("location"),
      null,
      "Location header should be null (not redirected)"
    );

    const researchRes = await fetch("http://localhost:3000/api/research/", {
      method: "POST",
      redirect: "manual",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.notEqual(
      researchRes.status,
      308,
      "/api/research/ should not return 308 Permanent Redirect"
    );
    assert.equal(
      researchRes.headers.get("location"),
      null,
      "Location header should be null (not redirected)"
    );
  } catch (err) {
    if (err.cause?.code === "ECONNREFUSED" || err.code === "ECONNREFUSED") {
      t.skip("Next.js dev server not running on port 3000, skipping live HTTP check");
    } else {
      throw err;
    }
  }
});
