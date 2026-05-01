const form = document.getElementById("form");
const resultDiv = document.getElementById("result");
const loading = document.getElementById("loading");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // 👉 Validate cơ bản
    if (!credit_score.value || !age.value || !balance.value) {
        alert("Please fill all required fields!");
        return;
    }

    // 👉 Build data gửi API
    const data = {
        credit_score: parseInt(credit_score.value),
        country: country.value,
        gender: gender.value,
        age: parseInt(age.value),
        tenure: 3,
        balance: parseFloat(balance.value),
        products_number: parseInt(products_number.value),
        credit_card: 1,
        active_member: parseInt(active_member.value),
        estimated_salary: 50000
    };

    // 👉 Show loading
    loading.classList.remove("hidden");
    resultDiv.innerHTML = "";

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await res.json();

        // 👉 xử lý kết quả
        const isChurn = result.churn === 1;
        const prob = (result.probability * 100).toFixed(1);

        // 👉 hiển thị đẹp
        resultDiv.innerHTML = `
            <div class="result-box ${isChurn ? "churn" : "stay"}">
                <h3>${isChurn ? "⚠️ High Risk Customer" : "✅ Safe Customer"}</h3>
                <p>Churn Probability: <b>${prob}%</b></p>
            </div>
        `;

    } catch (error) {
        console.error(error);

        resultDiv.innerHTML = `
            <div class="result-box churn">
                <h3>❌ Error</h3>
                <p>Something went wrong. Please try again.</p>
            </div>
        `;
    }

    // 👉 hide loading
    loading.classList.add("hidden");
});