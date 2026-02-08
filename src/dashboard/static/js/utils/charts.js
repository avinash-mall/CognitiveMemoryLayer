/**
 * Chart.js helper wrappers for the dashboard.
 * Provides consistent styling and theme-aware colors.
 */

/** Get current theme colors from CSS variables */
function getThemeColors() {
    const style = getComputedStyle(document.documentElement);
    return {
        text: style.getPropertyValue('--text-secondary').trim(),
        textMuted: style.getPropertyValue('--text-muted').trim(),
        border: style.getPropertyValue('--border').trim(),
        accent: style.getPropertyValue('--accent').trim(),
        success: style.getPropertyValue('--success').trim(),
        warning: style.getPropertyValue('--warning').trim(),
        danger: style.getPropertyValue('--danger').trim(),
        info: style.getPropertyValue('--info').trim(),
        purple: style.getPropertyValue('--purple').trim(),
        pink: style.getPropertyValue('--pink').trim(),
        teal: style.getPropertyValue('--teal').trim(),
        orange: style.getPropertyValue('--orange').trim(),
    };
}

/** Palette of chart colors */
export function getChartPalette() {
    const c = getThemeColors();
    return [
        c.accent, c.success, c.warning, c.danger, c.info,
        c.purple, c.pink, c.teal, c.orange,
        '#94a3b8', '#e879f9', '#22d3ee', '#a3e635', '#f97316',
        '#818cf8', '#fb7185',
    ];
}

/** Default chart options with theme-aware styling */
function baseOptions(title = '') {
    const c = getThemeColors();
    return {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            title: {
                display: !!title,
                text: title,
                color: c.text,
                font: { size: 13, weight: '600' },
                padding: { bottom: 12 },
            },
            legend: {
                labels: {
                    color: c.text,
                    font: { size: 11 },
                    padding: 14,
                    usePointStyle: true,
                    pointStyleWidth: 10,
                },
            },
            tooltip: {
                backgroundColor: 'rgba(20, 24, 40, 0.95)',
                titleColor: '#e4e7f0',
                bodyColor: '#9197b3',
                borderColor: 'rgba(50, 56, 85, 0.8)',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 10,
            },
        },
        scales: {},
    };
}

/** Destroy an existing chart stored on a canvas element */
function destroyExisting(canvas) {
    if (canvas._chartInstance) {
        canvas._chartInstance.destroy();
        canvas._chartInstance = null;
    }
}

/** Create a doughnut chart */
export function createDoughnutChart(canvas, labels, data, title = '') {
    destroyExisting(canvas);
    const palette = getChartPalette();
    const chart = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: palette.slice(0, data.length),
                borderWidth: 0,
                hoverOffset: 6,
            }],
        },
        options: {
            ...baseOptions(title),
            cutout: '65%',
            plugins: {
                ...baseOptions(title).plugins,
                legend: {
                    ...baseOptions(title).plugins.legend,
                    position: 'right',
                },
            },
        },
    });
    canvas._chartInstance = chart;
    return chart;
}

/** Create a bar chart */
export function createBarChart(canvas, labels, data, title = '', color = null) {
    destroyExisting(canvas);
    const c = getThemeColors();
    const palette = getChartPalette();
    const bgColors = color ? Array(data.length).fill(color) : palette.slice(0, data.length);

    const chart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: bgColors,
                borderRadius: 4,
                maxBarThickness: 50,
            }],
        },
        options: {
            ...baseOptions(title),
            plugins: {
                ...baseOptions(title).plugins,
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: { color: c.textMuted, font: { size: 11 } },
                    grid: { display: false },
                },
                y: {
                    ticks: { color: c.textMuted, font: { size: 11 } },
                    grid: { color: c.border + '40' },
                    beginAtZero: true,
                },
            },
        },
    });
    canvas._chartInstance = chart;
    return chart;
}

/** Create a line chart */
export function createLineChart(canvas, labels, data, title = '', color = null) {
    destroyExisting(canvas);
    const c = getThemeColors();
    const lineColor = color || c.accent;

    const chart = new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data,
                borderColor: lineColor,
                backgroundColor: lineColor + '20',
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                pointHoverRadius: 6,
                pointBackgroundColor: lineColor,
                borderWidth: 2,
            }],
        },
        options: {
            ...baseOptions(title),
            plugins: {
                ...baseOptions(title).plugins,
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: { color: c.textMuted, font: { size: 10 }, maxTicksLimit: 12 },
                    grid: { display: false },
                },
                y: {
                    ticks: { color: c.textMuted, font: { size: 11 } },
                    grid: { color: c.border + '40' },
                    beginAtZero: true,
                },
            },
        },
    });
    canvas._chartInstance = chart;
    return chart;
}
