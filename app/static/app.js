const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const answer = document.getElementById("answer");
const details = document.getElementById("details");
const anomalyButton = document.getElementById("anomaly-button");
const anomalySummary = document.getElementById("anomaly-summary");
const anomalyTable = document.getElementById("anomaly-table");

askButton.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) return;
  answer.textContent = "Thinking...";
  details.textContent = "";
  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Request failed");
    answer.textContent = data.answer;
    details.textContent = JSON.stringify(data.interpreted_query, null, 2);
  } catch (error) {
    answer.textContent = `Error: ${error.message}`;
  }
});

anomalyButton.addEventListener("click", async () => {
  anomalySummary.textContent = "Scanning...";
  anomalyTable.innerHTML = "";
  try {
    const response = await fetch("/anomalies");
    const data = await response.json();
    anomalySummary.textContent = `${data.summary.total_anomalies} anomalies found. Long-resolution threshold: ${data.long_resolution_threshold_hrs} hours.`;
    data.anomalies.forEach((item) => {
      const row = document.createElement("tr");
      row.innerHTML = `<td>${item.ticket_id}</td><td>${item.type}</td><td>${item.priority}</td><td>${item.status}</td><td>${item.reason}</td>`;
      anomalyTable.appendChild(row);
    });
  } catch (error) {
    anomalySummary.textContent = `Error: ${error.message}`;
  }
});
