// Verify the WASM artifact matches the Python reference (~1964 s).
import { teq_main_js } from "./pkg/sfeprapy_wasm.js";

const kwargs = {
    beamCrossSectionArea: 0.017, beamRho: 7850,
    beamPositionVertical: 2.5, beamPositionHorizontal: 18,
    fireTimeStep: 1.0, fireTimeDuration: 5 * 60 * 60,
    fireCombustionEfficiency: 0.8, fireHrrDensity: 0.25,
    fireLoadDensity: 420, fireMode: 0, fireNftLimit: 1050,
    fireSpreadSpeed: 0.01, fireTlim: 0.333,
    protectionC: 1700, protectionK: 0.2,
    protectionProtectedPerimeter: 2.14, protectionRho: 800,
    roomBreadth: 16, roomDepth: 31.25, roomHeight: 3,
    roomWallThermalInertia: 720,
    solverTemperatureGoal: 620 + 273.15, solverTol: 0.01,
    windowHeight: 2, windowWidth: 57.6,
    timberBurningRate: 0, timberSolverIlim: 20, timberSolverTol: 1,
};

const r = teq_main_js(kwargs);
const teq = r.solverTimeEquivalenceSolved;
console.log("WASM teq =", teq.toFixed(3), "s  (", (teq / 60).toFixed(1), "min )");
console.log("fire_type =", r.fireType, "| protection_thickness =", r.solverProtectionThickness);

const ok = Math.abs(teq - 1964) < 5;
console.log("\nfidelity check (expect ~1964 s):", ok ? "PASS" : `FAIL (got ${teq})`);
if (!ok) process.exit(1);

// Quick throughput: 1000 calls, measure ms.
const t0 = process.hrtime.bigint();
for (let i = 0; i < 1000; i++) teq_main_js(kwargs);
const dt = Number(process.hrtime.bigint() - t0) / 1e6;
console.log(`\nthroughput: ${dt / 1000} ms/call (1000 calls in ${dt} ms)`);
