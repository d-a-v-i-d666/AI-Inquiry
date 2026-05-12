import { useEffect, useMemo, useState } from "react";
import "./App.css";

type Candidate = {
    idx: number;
    department: string;
    confidence: number;
};

type PredictResponse = {
    input: string;
    processed_input: string;
    top1: Candidate;
    topk: Candidate[];
};

type QwenResponse = {
    input: string;
    processed_input: string;
    department?: string | null;
    confidence?: number | null;
    reason?: string | null;
    raw_output: string;
    parsed: boolean;
};

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

type HealthStatus = "unknown" | "ok" | "down";
type ModelMode = "classifier" | "qwen";

export default function App() {
    const [text, setText] = useState("");
    const [topK, setTopK] = useState(5);
    const [maxNewTokens, setMaxNewTokens] = useState(128);
    const [result, setResult] = useState<PredictResponse | null>(null);
    const [qwenResult, setQwenResult] = useState<QwenResponse | null>(null);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const [health, setHealth] = useState<HealthStatus>("unknown");
    const [mode, setMode] = useState<ModelMode>("classifier");

    const statusLabel = useMemo(() => {
        if (health === "ok") return "后端可用";
        if (health === "down") return "后端不可用";
        return "检测中";
    }, [health]);

    useEffect(() => {
        let alive = true;
        fetch(`${API_BASE}/healthz`)
            .then((res) => {
                if (!alive) return;
                setHealth(res.ok ? "ok" : "down");
            })
            .catch(() => {
                if (!alive) return;
                setHealth("down");
            });
        return () => {
            alive = false;
        };
    }, []);

    const handleSubmit = async () => {
        const trimmed = text.trim();
        if (!trimmed) {
            setError("请输入症状描述。");
            return;
        }

        setLoading(true);
        setError("");
        setResult(null);
        setQwenResult(null);

        try {
            const res = await fetch(
                mode === "classifier" ? `${API_BASE}/predict` : `${API_BASE}/qwen/predict`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(
                        mode === "classifier"
                            ? {
                                text: trimmed,
                                top_k: topK,
                            }
                            : {
                                text: trimmed,
                                max_new_tokens: maxNewTokens,
                            }
                    ),
                }
            );

            if (!res.ok) {
                const payload = await res.json().catch(() => null);
                const message = payload?.detail || `请求失败: ${res.status}`;
                throw new Error(message);
            }

            if (mode === "classifier") {
                const payload = (await res.json()) as PredictResponse;
                setResult(payload);
            } else {
                const payload = (await res.json()) as QwenResponse;
                setQwenResult(payload);
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : "请求失败";
            setError(message);
        } finally {
            setLoading(false);
        }
    };

    const handleExample = (value: string) => {
        setText(value);
        setResult(null);
        setQwenResult(null);
        setError("");
    };

    return (
        <div className="app">
            <div className="hero">
                <div className="hero-text">
                    <span className="pill">AI 分诊助手</span>
                    <h1>快速判断就诊科室</h1>
                    <p>
                        输入你的症状描述，模型将给出推荐科室与 Top-K 备选结果，
                        适用于门诊初筛与线上问诊场景。
                    </p>
                    <div className="status">
                        <span className={`status-dot ${health}`}></span>
                        <span>{statusLabel}</span>
                    </div>
                </div>
                <div className="hero-card">
                    <h2>示例输入</h2>
                    <button onClick={() => handleExample("最近咳嗽发烧胸闷，应该挂什么科？")}
                        type="button">
                        咳嗽发烧胸闷
                    </button>
                    <button onClick={() => handleExample("关节反复疼痛肿胀，晨僵明显")}
                        type="button">
                        关节疼痛晨僵
                    </button>
                    <button onClick={() => handleExample("腹痛伴恶心，食欲下降")}
                        type="button">
                        腹痛恶心
                    </button>
                </div>
            </div>

            <div className="panel">
                <div className="panel-header">
                    <h2>输入描述</h2>
                    <div className="controls">
                        <div className="mode-toggle">
                            <button
                                type="button"
                                className={mode === "classifier" ? "mode-btn active" : "mode-btn"}
                                onClick={() => setMode("classifier")}
                            >
                                MC-BERT
                            </button>
                            <button
                                type="button"
                                className={mode === "qwen" ? "mode-btn active" : "mode-btn"}
                                onClick={() => setMode("qwen")}
                            >
                                Qwen2.5
                            </button>
                        </div>
                        {mode === "classifier" ? (
                            <div className="slider">
                                <label htmlFor="topk">Top-K</label>
                                <input
                                    id="topk"
                                    type="range"
                                    min={1}
                                    max={10}
                                    value={topK}
                                    onChange={(e) => setTopK(Number(e.target.value))}
                                />
                                <span>{topK}</span>
                            </div>
                        ) : (
                            <div className="slider">
                                <label htmlFor="tokens">Max tokens</label>
                                <input
                                    id="tokens"
                                    type="range"
                                    min={32}
                                    max={256}
                                    step={8}
                                    value={maxNewTokens}
                                    onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                                />
                                <span>{maxNewTokens}</span>
                            </div>
                        )}
                    </div>
                </div>
                <textarea
                    placeholder="例如：最近咳嗽发烧胸闷，应该挂什么科？"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    rows={5}
                />
                <div className="actions">
                    <button className="primary" onClick={handleSubmit} disabled={loading}>
                        {loading ? "分析中..." : "生成结果"}
                    </button>
                    <button
                        className="ghost"
                        onClick={() => {
                            setText("");
                            setResult(null);
                            setQwenResult(null);
                            setError("");
                        }}
                        type="button"
                    >
                        清空
                    </button>
                </div>
                {error && <div className="alert">{error}</div>}
            </div>

            <div className="result">
                <h2>推荐结果</h2>
                {!result && !qwenResult && <div className="placeholder">等待模型返回结果</div>}
                {result && mode === "classifier" && (
                    <div className="result-grid">
                        <div className="result-card">
                            <h3>Top 1</h3>
                            <div className="dept">{result.top1.department}</div>
                            <div className="confidence">
                                置信度: {result.top1.confidence.toFixed(4)}
                            </div>
                            <div className="meta">处理后输入：{result.processed_input}</div>
                        </div>
                        <div className="result-card">
                            <h3>Top K 列表</h3>
                            <ul>
                                {result.topk.map((item, index) => (
                                    <li key={`${item.department}-${index}`}>
                                        <span>{index + 1}. {item.department}</span>
                                        <span>{item.confidence.toFixed(4)}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                )}
                {qwenResult && mode === "qwen" && (
                    <div className="result-grid">
                        <div className="result-card">
                            <h3>推荐科室</h3>
                            <div className="dept">
                                {qwenResult.department || "未解析"}
                            </div>
                            <div className="confidence">
                                置信度: {qwenResult.confidence?.toFixed(4) ?? "-"}
                            </div>
                            <div className="meta">理由：{qwenResult.reason || "-"}</div>
                            {!qwenResult.parsed && (
                                <div className="alert">JSON 解析失败，请检查输出格式。</div>
                            )}
                        </div>
                        <div className="result-card">
                            <h3>模型输出</h3>
                            <details>
                                <summary>查看原始输出</summary>
                                <pre className="raw-output">{qwenResult.raw_output}</pre>
                            </details>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
