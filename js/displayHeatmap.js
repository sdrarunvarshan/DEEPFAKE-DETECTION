// Select the canvas element
const canvas = document.getElementById("heatmap");

// Example data for the heatmap
const heatmapData = [
    { x: 1, y: 1, v: 0.5 },
    { x: 2, y: 1, v: 0.8 },
    { x: 3, y: 2, v: 0.2 },
    { x: 4, y: 3, v: 1.0 },
];

function displayHeatmap(canvas, data) {
    new Chart(canvas, {
        type: "matrix",
        data: {
            datasets: [
                {
                    label: "Heatmap",
                    data: data,
                    backgroundColor(ctx) {
                        const value = ctx.raw.v;
                        return `rgba(0, 0, 255, ${value})`;
                    },
                    borderWidth: 1,
                    width(ctx) {
                        const chart = ctx.chart;
                        if (chart.chartArea) {
                            return chart.chartArea.width / 10 - 4; // Adjust for padding
                        }
                        return 30; // Default width
                    },
                    height(ctx) {
                        const chart = ctx.chart;
                        if (chart.chartArea) {
                            return chart.chartArea.height / 10 - 4; // Adjust for padding
                        }
                        return 30; // Default height
                    },
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { type: "linear", position: "bottom" },
                y: { type: "linear" },
            },
            plugins: {
                legend: { display: false },
            },
        },
    });
}
