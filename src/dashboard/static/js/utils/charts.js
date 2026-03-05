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
        animation: {
            duration: 600,
            easing: 'easeOutQuart',
        },
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
                backgroundColor: 'rgba(15, 17, 23, 0.96)',
                titleColor: '#e4e7f0',
                bodyColor: '#9197b3',
                borderColor: 'rgba(108, 140, 255, 0.3)',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 12,
                titleFont: { weight: '600', size: 12 },
                bodyFont: { size: 11 },
                displayColors: true,
                boxPadding: 4,
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
    const opts = baseOptions(title);
    const chart = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: palette.slice(0, data.length),
                borderWidth: 0,
                hoverOffset: 8,
                hoverBorderWidth: 2,
                hoverBorderColor: 'rgba(255,255,255,0.3)',
            }],
        },
        options: {
            ...opts,
            cutout: '65%',
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 800,
                easing: 'easeOutQuart',
            },
            plugins: {
                ...opts.plugins,
                legend: {
                    ...opts.plugins.legend,
                    position: 'right',
                },
                tooltip: {
                    ...opts.plugins.tooltip,
                    callbacks: {
                        label: (ctx) => {
                            const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                            const pct = total > 0 ? ((ctx.parsed / total) * 100).toFixed(1) : '0';
                            return ` ${ctx.label}: ${ctx.parsed.toLocaleString()} (${pct}%)`;
                        },
                    },
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
    const hoverColors = bgColors.map(c => c + 'cc');

    const opts = baseOptions(title);
    const chart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: bgColors,
                hoverBackgroundColor: hoverColors,
                borderRadius: 6,
                maxBarThickness: 50,
                borderSkipped: false,
            }],
        },
        options: {
            ...opts,
            plugins: {
                ...opts.plugins,
                legend: { display: false },
                tooltip: {
                    ...opts.plugins.tooltip,
                    callbacks: {
                        label: (ctx) => ` ${ctx.label}: ${ctx.parsed.y.toLocaleString()}`,
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: c.textMuted, font: { size: 11 } },
                    grid: { display: false },
                },
                y: {
                    ticks: { color: c.textMuted, font: { size: 11 } },
                    grid: { color: c.border + '30', lineWidth: 0.8 },
                    beginAtZero: true,
                },
            },
        },
    });
    canvas._chartInstance = chart;
    return chart;
}

/** Create a line chart with gradient fill */
export function createLineChart(canvas, labels, data, title = '', color = null) {
    destroyExisting(canvas);
    const c = getThemeColors();
    const lineColor = color || c.accent;

    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.parentElement?.clientHeight || 280);
    gradient.addColorStop(0, lineColor + '40');
    gradient.addColorStop(0.6, lineColor + '10');
    gradient.addColorStop(1, lineColor + '00');

    const opts = baseOptions(title);
    const chart = new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data,
                borderColor: lineColor,
                backgroundColor: gradient,
                fill: true,
                tension: 0.35,
                pointRadius: 2,
                pointHoverRadius: 6,
                pointBackgroundColor: lineColor,
                pointBorderColor: 'transparent',
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2,
                borderWidth: 2.5,
            }],
        },
        options: {
            ...opts,
            plugins: {
                ...opts.plugins,
                legend: { display: false },
                tooltip: {
                    ...opts.plugins.tooltip,
                    intersect: false,
                    mode: 'index',
                },
            },
            scales: {
                x: {
                    ticks: { color: c.textMuted, font: { size: 10 }, maxTicksLimit: 12 },
                    grid: { display: false },
                },
                y: {
                    ticks: { color: c.textMuted, font: { size: 11 } },
                    grid: { color: c.border + '30', lineWidth: 0.8 },
                    beginAtZero: true,
                },
            },
        },
    });
    canvas._chartInstance = chart;
    return chart;
}
