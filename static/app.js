// static/app.js

const form      = document.getElementById("form");
const resultDiv = document.getElementById("result");
const loading   = document.getElementById("loading");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const data = {
        credit_score     : parseFloat(document.getElementById("credit_score").value),
        country          : document.getElementById("country").value,
        gender           : document.getElementById("gender").value,
        age              : parseFloat(document.getElementById("age").value),
        tenure           : parseFloat(document.getElementById("tenure").value),
        balance          : parseFloat(document.getElementById("balance").value),
        products_number  : parseInt(document.getElementById("products_number").value),
        credit_card      : parseInt(document.getElementById("credit_card").value),
        active_member    : parseInt(document.getElementById("active_member").value),
        estimated_salary : parseFloat(document.getElementById("estimated_salary").value),
    };

    // ── Validate ─────────────────────────────────
    for (const [key, val] of Object.entries(data)) {
        if (val === null || val === undefined || (typeof val === "number" && isNaN(val))) {
            alert(`Vui lòng nhập đầy đủ: ${key}`);
            return;
        }
    }

    loading.classList.remove("hidden");
    resultDiv.innerHTML = "";

    try {
        const res = await fetch("/predict", {
            method : "POST",
            headers: { "Content-Type": "application/json" },
            body   : JSON.stringify(data)
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `Server error: ${res.status}`);
        }

        const result = await res.json();

        if (result.churn_probability === undefined || isNaN(result.churn_probability)) {
            throw new Error("Invalid response from server");
        }

        const isChurn = result.churn_prediction === 1;
        const prob    = (result.churn_probability * 100).toFixed(1);
        const risk    = result.risk_level;

        // ── Nền TRẮNG + chữ màu → hiển thị rõ trên cả đỏ lẫn xanh ──
        const riskColor = { "CAO": "#d32f2f", "TRUNG": "#e65100", "THẤP": "#2e7d32" }[risk] || "#333";
        const riskBadge = `<span style="
            background: white;
            color: ${riskColor};
            padding: 2px 14px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 0.9rem;
        ">${risk}</span>`;

        resultDiv.innerHTML = `
            <div class="result-box ${isChurn ? "churn" : "stay"}">
                <h3>${isChurn ? "High Risk Customer" : "Safe Customer"}</h3>
                <hr/>
                <p>Churn Probability : <b>${prob}%</b></p>
                <p>Risk Level &nbsp;&nbsp;&nbsp;&nbsp;: ${riskBadge}</p>
                <p>Threshold &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:
                   <b>${(result.threshold_used * 100).toFixed(0)}%</b></p>
            </div>
        `;

    } catch (err) {
        console.error(err);
        resultDiv.innerHTML = `
            <div class="result-box churn">
                <h3>Error</h3>
                <p>${err.message}</p>
            </div>
        `;
    }

    loading.classList.add("hidden");
});