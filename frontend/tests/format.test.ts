import assert from "node:assert/strict";
import { test } from "node:test";
import { machineToDraft, niceTicks, pct, scale, signedPoints, validateMachine } from "../src/lib/format";

const valid = {
  type: "L", air_temp_k: "300", process_temp_k: "310", rotational_speed_rpm: "1500", torque_nm: "40", tool_wear_min: "100",
};

test("valid draft becomes a typed machine", () => {
  const { machine, errors } = validateMachine(valid);
  assert.deepEqual(errors, []);
  assert.deepEqual(machine, { type: "L", air_temp_k: 300, process_temp_k: 310, rotational_speed_rpm: 1500, torque_nm: 40, tool_wear_min: 100 });
});

test("empty field is reported", () => {
  const { machine, errors } = validateMachine({ ...valid, torque_nm: "" });
  assert.equal(machine, null);
  assert.equal(errors[0]?.field, "torque_nm");
});

test("non-numeric and out-of-range values are reported", () => {
  const { errors } = validateMachine({ ...valid, torque_nm: "forty", rotational_speed_rpm: "-1200" });
  const fields = errors.map((e) => e.field).sort();
  assert.deepEqual(fields, ["rotational_speed_rpm", "torque_nm"]);
  assert.match(errors.find((e) => e.field === "torque_nm")?.message ?? "", /number/);
  assert.match(errors.find((e) => e.field === "rotational_speed_rpm")?.message ?? "", /between/);
});

test("invalid product type is reported", () => {
  const { errors } = validateMachine({ ...valid, type: "Q" });
  assert.equal(errors[0]?.field, "type");
});

test("machineToDraft round-trips", () => {
  const draft = machineToDraft({ type: "M", torque_nm: 12.5 });
  assert.deepEqual(draft, { type: "M", torque_nm: "12.5" });
});

test("formatting helpers", () => {
  assert.equal(pct(0.1234), "12.3%");
  assert.equal(pct(null), "—");
  assert.equal(signedPoints(0.052), "+5.2 pts");
  assert.equal(signedPoints(-0.4), "−40.0 pts");
  assert.equal(signedPoints(0), "±0.0 pts");
});

test("scale and ticks", () => {
  const s = scale(0, 10, 0, 100);
  assert.equal(s(5), 50);
  assert.deepEqual(niceTicks(0, 1, 4), [0, 0.25, 0.5, 0.75, 1]);
  assert.deepEqual(niceTicks(0, 100, 5), [0, 20, 40, 60, 80, 100]);
});
