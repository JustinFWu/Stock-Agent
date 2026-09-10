##################################################################
Human:

Dead code
Almost none, which is rare.
    •    SECTOR_MAP — 82 entries mapping tickers to sector ETFs, used only as tuple(sorted(SECTOR_MAP)) to produce the ticker list. The sector values are never read. It's documented as feeding "per-sector exposure limits" for the risk layer, and that layer doesn't exist yet. Not junk, but it's a promise, not a feature.
    •    PricePanel.slice_dates — defined, never called anywhere including tests.
    •    Portfolio.weights and exit_removed are used. predict_vol is unused outside its module but that's the Phase 4 seam.
    •    .claude/skills, .github/workflows are live.
 
The Phase 2 backtester is excellent and heavily verified, but three things stand out. The strategy interface is scale-ambiguous: the same conviction shape yields 70% or 93% invested depending on arbitrary magnitude, and Phase 3's vol targeting emits exactly the raw scores that trip it. A name halted on execution day is dropped silently, not deferred to the next session. The real weak spot is coverage: all 47 tests are Phase 2, leaving the entire ~780-line Phase 1 model stack untested — including the smearing correction that sizes every position 9% light if bypassed. CI runs lint only, and the parity gate can't run without local cache.

###################################################################
Agent:

Review completed 2026-09-05. Reviewed all 38 tracked files, plus this existing review: application modules, configuration, CLI, tests, CI, documentation (including the roadmap's CSS/JavaScript), and repository-local agent/skill instructions. Generated caches, saved models, dependencies, and Git internals were excluded from source review. Existing cached bars were used by the test suite. Findings below describe the current code; application code was not changed by this review. Concurrent edits observed at completion removed `PricePanel.slice_dates` and clarified the unused sector values; those edits were inspected and left intact.

Priority: P1 = correctness issue to fix before relying on affected results; P2 = reliability or validation gap; P3 = lower-impact robustness or hygiene.

1. **[P1] Opening-time valuation can read the execution day's future close.**

   Location: `stock-agent/src/backtest/engine.py:138`, especially line 143.

   Issue: `marks = panel.closes.ffill()` includes today's close. `panel.opens.combine_first(marks)` therefore fills a missing opening price with today's close when that close exists, even though trades are sized at the open. This contaminates NAV and trades in other holdings.

   Evidence: In a four-day, two-name, zero-cost backtest with 50/50 targets, making BBB's opening price missing and changing only its subsequent close from 100 to 50 changes the same morning's AAA trade from no fill to a sale of 125 shares. The opening information and previous targets are identical.

   Suggest: Fall back to the last close strictly before the execution session, e.g. forward-filled closes shifted by one session. Add a regression asserting that changing any execution-day close cannot change that morning's orders.

2. **[P1] Minimum commissions still let buy scaling borrow cash.**

   Location: `stock-agent/src/backtest/portfolio.py:190`; `stock-agent/src/backtest/costs.py:72`.

   Issue: Scaling shares by `available / required` assumes every component of required cash scales proportionally or faster. A minimum commission does neither. The final fill recalculates the same fixed fee after reducing the notional, exceeding the reserved cash. The existing cash-scaling test uses a per-share commission but no minimum fee.

   Evidence: With 100 cash, an order for two shares at 100, zero spread/impact, and `min_commission=1`, `execute()` leaves cash at **-0.5024875621890601**. This disproves the comment that the estimate is conservative for all supported cost configurations.

   Suggest: Solve affordability using fees recomputed at the scaled quantities, reserving fixed fees or using a bounded search. Skip orders when cash cannot cover their minimum fees, and verify nonnegative cash for multiple simultaneous buys.

3. **[P1] Derived caches can silently survive changes to their underlying bars and labels.**

   Location: `stock-agent/src/data/dataset.py:65`; `stock-agent/src/data/fetcher.py:98`.

   Issue: A feature cache is trusted whenever the raw manifest matches the current start date and interval. It records no dependency on the particular raw-data version, feature configuration, label horizon, or code version. Running `--fetch --refetch` and then training without `--rebuild` reuses the older derived frame. Changing `HISTORY_START` and fetching first also makes the manifest look current before the old feature cache is checked. Changing `FORWARD_DAYS` can train on old labels while the saved payload advertises the new horizon.

   Evidence: A mocked cache-path check with a matching raw manifest returned the old cached frame and made zero calls to `build_ticker_frame`. Inspection confirms no other invalidation or cache metadata exists.

   Suggest: Fingerprint the actual raw input, relevant configuration, and feature/label schema in every derived cache. Rebuild when any dependency changes; invalidate derived caches when raw data is replaced.

4. **[P1] The walk-forward embargo is insufficient for tickers with missing sessions.**

   Location: `stock-agent/src/models/vol_forecast.py:75`; `stock-agent/src/labels/target.py:46`.

   Issue: Labels look ahead five observed rows of an individual ticker, while the embargo removes five dates from the pooled calendar. If a ticker has a gap, its fifth subsequent observation can lie inside the test block even when its feature date survives the embargo. Training then uses outcomes from the evaluation period.

   Evidence: On a 20-business-day pooled calendar, a ticker observed on indices 0-4 and 10-19 retains its index-4 training row with a five-date embargo. That row's label ends at index 14, inside the test block beginning at index 10. Reproduced with the actual label builder and `_date_windows`.

   Suggest: Carry the actual label-end timestamp with each training row and purge rows whose outcome window reaches the test block. Alternatively, align labels to a common session calendar and reject incomplete windows. Test gaps as well as late IPOs.

5. **[P2] The live wrapper accepts indefinitely stale data.**

   Location: `stock-agent/src/data/fetcher.py:98`; `stock-agent/src/strategy/live.py:56`; `stock-agent/pipeline.py:42`.

   Issue: `is_current` deliberately ignores the last-bar date and claims freshness is checked where live signals are formed. The live wrapper instead defaults to `panel.dates[-1]` and only verifies that the chosen date exists. An old cache therefore produces apparently valid live weights indefinitely; routine `--fetch` also skips it. Partial freshness is problematic too: one updated ticker advances the panel date while stale names disappear from the candidates.

   Suggest: Distinguish cache-configuration compatibility from freshness. Require the expected completed session and sufficient per-name coverage for live calls, with explicit historical replay behavior. Reject incomplete current-session bars. This is a missing guard at the existing live-weight seam; broker execution remains a future phase.

6. **[P2] A missed execution is discarded rather than deferred or reported.**

   Location: `stock-agent/src/backtest/portfolio.py:139`; `stock-agent/src/backtest/engine.py:163`.

   Issue: `plan_trades` skips a ticker lacking an executable open, and the engine clears the entire pending target immediately afterward. If a removed holding cannot be sold that morning, it can remain held until the next scheduled rebalance despite resuming trading the following day. There is no unfilled-order record or caveat. This confirms the Human section's observation.

   Suggest: Define an explicit missed-execution policy. Retain unresolved intent until the next tradable session, subject to cancellation by newer targets, and record deferrals/rejections. Test a missed liquidation followed by recovery on a non-rebalance day.

7. **[P2] Model preparation does not enforce finite values or a complete required schema.**

   Location: `stock-agent/src/models/vol_forecast.py:62`, `:191`, `:364`.

   Issue: `dropna` does not reject infinities, and only the label is checked for positivity even though HAR takes logs of its inputs. `get_available_features` silently accepts a reduced feature set; `prepare` requires HAR columns but not `ewma_vol`, which the mandatory EWMA forecaster accesses unconditionally. A missing feature can therefore alter the fitted model silently, while a missing baseline column or invalid value causes failure later in validation. Prediction applies log transforms without equivalent validation.

   Evidence: `prepare` retained an infinite `ewma_vol` in a targeted check. Zero HAR inputs likewise survive preparation and become negative infinity under the subsequent log.

   Suggest: Validate the required schema, finite numeric values, and positive log inputs at the boundary, with clear errors or explicit row rejection counts. Make any reduced feature set an intentional versioned configuration and apply the saved schema at inference.

8. **[P2] Invalid risk and cost settings can violate the promised constraints.**

   Location: `stock-agent/src/strategy/weights.py:98`; `stock-agent/src/backtest/costs.py:42`; `stock-agent/src/backtest/engine.py:104`.

   Issue: Public parameters are not validated. A negative `max_weight` or `max_gross` turns positive proposals into negative output weights after the input sanitization, allowing a supposedly long-only path to propose shorts. NaN limits can bypass comparisons. Negative cost parameters can reward trading, and nonpositive initial cash produces meaningless runs.

   Evidence: `_apply_constraints(Series({'AAA': 1}), ['AAA'], -0.1, 1)` returns `{'AAA': -0.1}`.

   Suggest: Validate finite, domain-appropriate limits at public entry points and cost-model construction. Permit intentional zero exposure, reject negative exposure/cost limits, and check output invariants after constraints are applied.

9. **[P2] CI does not execute the test suite, and Phase 1 has no automated regression coverage.**

   Location: `.github/workflows/ci.yml:3`; `stock-agent/tests/`.

   Issue: CI still says to add pytest once tests exist. It runs Ruff and byte compilation only on `stock-agent/src`, omitting `pipeline.py`, `config.py`, and tests. The 47 tests cover Phase 2; none exercise fetching, feature caches, forward labels, model preparation, embargo behavior, smearing, or training/prediction parity. Thus the reproduced Phase 1 bugs do not affect the green suite.

   Suggest: Run deterministic pytest tests in CI and lint all authored Python. Prioritize label alignment, ragged embargo purging, cache invalidation, and saved-model prediction parity. Resolve finding 10 before relying on a clean-checkout test job.

10. **[P2] The parity fixture errors on an empty cache before it can skip.**

    Location: `stock-agent/tests/test_weight_parity.py:99`; `stock-agent/src/data/panel.py:136`.

    Issue: `real_panel` calls `load_price_panel` before checking whether fewer than ten tickers loaded. With no cached files, the loader raises `ValueError`, so the advertised skip is unreachable and the four parity tests error on a clean checkout. Even a successful skip would leave the principal parity gate untested there.

    Evidence: Mocking all bar reads as missing reproduced the loader's `ValueError`. The fixture has no handler around that call.

    Suggest: Make the gate operate on a deterministic, history-sensitive synthetic panel with an independently truncated live counterpart. Keep real-data coverage as a separately marked optional integration test, with explicit handling of absent cache and date coverage.

11. **[P2] Partial failures are not carried into machine-readable outcomes.**

    Location: `stock-agent/pipeline.py:35`; `stock-agent/src/data/dataset.py:73`; `stock-agent/src/data/panel.py:127`.

    Issue: Fetch failures are printed but the CLI exits successfully even if every download fails. Dataset assembly catches every `Exception` from feature/label construction, including programming errors, and can train on the surviving subset. Missing panel files are also only printed. The returned data, saved model, and backtest caveats do not preserve a complete requested-versus-loaded ticker report, so logs are the only evidence that coverage changed. Cached-frame reads and writes sit outside the dataset's error handler, making its advertised skip behavior inconsistent.

    Suggest: Return structured coverage/failure metadata, persist it with results, and enforce an explicit minimum coverage policy. Fail the CLI on total failure. Catch expected data-access errors separately from feature-code defects and surface the latter with their traceback.

12. **[P2] Relative reporting overstates what a same-universe baseline corrects.**

    Location: `stock-agent/src/backtest/metrics.py:110`, `:165`; `stock-agent/src/data/universe.py:26`; `docs/phase2-backtester.md:162`.

    Issue: The formatter calls the information ratio the "bias-cancelling number", and the surrounding documentation presents the same-universe subtraction as cancellation of most survivorship bias. The code merely subtracts daily returns. Shared constituents do not imply a shared additive bias: active return depends on different strategy/baseline weights, and omitting a failed name can affect their selections and weights differently. The synthetic test explicitly injects an identical additive component, so it establishes only that special case. The RSP comparison also changes constituent count/composition and rebalance implementation; its full performance gap is not an isolated measurement of survivorship bias.

    Suggest: Retain the useful baseline comparison but label it active performance on a survivor-selected universe. Carry the unresolved selection limitation into the relative output and avoid treating the RSP gap as a measured correction or proven upper bound. Re-evaluate selection and both portfolios on point-in-time data before drawing that conclusion.

13. **[P2] Strategy proposals mix relative scores and absolute exposure without an explicit contract.**

    Location: `stock-agent/src/strategy/weights.py:43`, `:103`.

    Issue: The protocol calls proposals relative weights, while values summing below one encode absolute cash allocation. Scaling a relative score vector can therefore change gross exposure. For example, with a 0.4 name cap, `(0.8, 0.4, 0.2)` produces gross approximately 0.82857, while the identical shape `(0.4, 0.2, 0.1)` produces 0.7. The current behavior is documented in pieces and intentional cash preservation is tested, but a future raw-score caller cannot state its intended exposure separately. This confirms the Human section's interface concern; it is not a claim that today's equal-weight strategy is mis-sized.

    Suggest: Require portfolio-fraction weights at the protocol boundary, or return normalized conviction and an explicit gross target as separate values. Test scale invariance for conviction and preservation of deliberate cash independently.

14. **[P3] A perfect baseline crashes gate reporting.**

    Location: `stock-agent/src/models/vol_forecast.py:295`.

    Issue: The percentage-margin calculation divides by the baseline score. QLIKE and RMSE can legitimately equal zero for a perfect forecast, so a valid result can crash instead of reporting that XGBoost failed to beat the baseline.

    Evidence: Supplying zero baseline losses to `_report_gate` raises `ZeroDivisionError: float division by zero`.

    Suggest: Handle zero baseline scores explicitly, retain the strict pass/fail comparison, and report an absolute improvement or an undefined percentage. Validate split counts too: `n_splits=-1` currently reaches division by zero in `_date_windows`.

15. **[P2] Cache and model updates overwrite usable artifacts in place.**

    Location: `stock-agent/src/data/fetcher.py:46`, `:82`; `stock-agent/src/data/dataset.py:80`; `stock-agent/src/models/vol_forecast.py:349`.

    Issue: Parquet, manifest JSON, and the sole saved model are written directly to their final paths. An interrupted write can destroy the previous usable artifact, and another reader can observe a partial file. The manifest uses an unlocked read-modify-write cycle, so overlapping fetch processes can also lose each other's entries. This is a persistence failure mode identified by inspection, not an induced corruption test.

    Suggest: Write complete artifacts to temporary files in the same directory, validate them, then atomically replace the destination. Coordinate manifest updates and tie raw/derived versions together so an interruption cannot leave a current-looking manifest attached to inconsistent data.

16. **[P3] Import bootstrapping and environment setup are duplicated and fragile.**

    Location: `stock-agent/src/data/dataset.py:20` and the equivalent `sys.path.append` in most modules/tests; `stock-agent/requirements.txt:1`; `README.md:1`.

    Issue: Modules mutate the process-wide import path to find a generic top-level `config` and `src` package. Appending the path does not prevent an earlier unrelated module from winning import resolution. The same workaround is repeated throughout the repository. Requirements contain only open-ended lower bounds, there is no recorded reproducible dependency environment, and the README gives no Python version, installation, or test instructions.

    Suggest: Package the application with an explicit entry point and package-local configuration, install it in editable mode for development, and remove per-module path mutation. Record a tested dependency set and document setup and test commands. Keep secret values outside those instructions.

17. **[P3] Documentation and comments contain misleading or stale statements.**

    Locations and concrete cleanups:

    - `README.md:1` describes an IBKR trading agent, whereas the implemented application is a research/backtest pipeline and the roadmap chooses Alpaca first. Describe the implemented scope and link to the roadmap.
    - `docs/roadmap.html:862` leaves the Phase 2 gate pending despite the completed Phase 2 record. It also refers to nonexistent `train.py` at lines 684 and 918. Distinguish historical plans from current implementation and update stale references.
    - `stock-agent/src/models/vol_forecast.py:341` and `docs/roadmap.html:834` say omitting smearing sizes positions about 9% light. For inverse-volatility sizing, underpredicting volatility makes positions larger: a correction factor of 1.099 implies roughly 9.9% more exposure if omitted, before caps. Clarify forecast bias versus sizing bias. The same wording in the Human section was preserved as requested.
    - `stock-agent/src/data/universe.py:48` calls the recorded 16.8% result buy-and-hold, while `docs/phase2-backtester.md:154` describes the measurement as daily rebalanced. Use the actual measurement's name consistently. The universe module also refers to nonexistent `metrics.relative_summary`; the function is `summarize_relative`.
    - `stock-agent/tests/test_engine.py:72` describes reconciliation on every day, but the test reconstructs only final NAV. Extend the assertion or narrow the claim. `tests/test_metrics.py:44` still names average-NAV cost drag although the implementation now uses each day's NAV.
    - `stock-agent/pipeline.py:153` is unreachable because `parser.error` already exits. Remove the redundant exit and its now-unneeded import when cleaning this up.
    - `.claude/skills/code-hygiene/SKILL.md` cites absent `news/sentiment.py` as a current example. Refresh repository-specific guidance when retired modules are removed.

    Preserve comments explaining time ordering, accounting units, and non-obvious tradeoffs. The highest-value hygiene change is correcting statements that assert a guarantee the implementation does not provide, rather than shortening comments indiscriminately.

Verification performed:

- Existing suite: `stock-agent/venv/Scripts/python.exe -m pytest -q` from `stock-agent` — **47 passed in 15.20s**, including the real-cache parity tests.
- Static check: `python -m pyflakes stock-agent/src stock-agent/tests stock-agent/config.py stock-agent/pipeline.py` using the repository venv — **passed**. Ruff was not installed in this environment and was not run.
- Environment: Python **3.13.1**, pandas **3.0.3**, NumPy **2.4.6**. CI specifies Python 3.12; that separate environment was not executed.
- Targeted in-memory checks reproduced the opening-time leak, negative cash under minimum fees, ragged-calendar label overlap, stale derived-cache reuse, retained infinity, negative constrained weights, empty-cache loader failure, and zero-loss reporting crash. These checks did not alter market-data caches or saved models.
- Full model training, fresh network downloads, and broker operations were not run. Historical numerical claims in the roadmap were not independently re-estimated.

Summary: **17 findings: 4 P1, 10 P2, and 3 P3.** Fix time ordering, fixed-fee affordability, and cache provenance first; add deterministic regression coverage for those failures and Phase 1 before relying on the existing green suite. No source fixes were applied as part of this review.

I love hot dogs
