function formatDate(unixEpoch) {
  const date = new Date(unixEpoch * 1000);
  const localDateTime = date.toLocaleString();
  return localDateTime;
}

function updateScorecards(tableRows) {
  let positiveCount = 0;
  let negativeCount = 0;
  let visibleRows = 0;

  tableRows.forEach(row => {
    if (row.style.display !== "none") {
      visibleRows++;
      const sentiment = row.querySelector("td:nth-child(3)").innerText;
      sentiment === "POSITIVE" ? positiveCount++ : sentiment === "NEGATIVE" ? negativeCount++ : null;
    }
  });

  const positiveRatio = positiveCount / visibleRows;
  const negativeRatio = negativeCount / visibleRows;
  const sentimentScore = (positiveRatio - negativeRatio).toFixed(2);

  document.getElementById("positive-ratio").innerText = `Positive Ratio: ${(positiveRatio * 100).toFixed(2)}%`;
  document.getElementById("negative-ratio").innerText = `Negative Ratio: ${(negativeRatio * 100).toFixed(2)}%`;
  document.getElementById("sentiment-score").innerText = `Sentiment Score: ${sentimentScore}`;
}


function createTableHeader(headerText) {
  const headerCell = document.createElement('th');
  headerCell.innerText = headerText;
  return headerCell;
}

function createTableCell(content, s=0) {
    const cell = document.createElement('td');
    if (s === 1) {cell.style.whiteSpace = 'nowrap';}

  cell.innerText = content;
  return cell;
}

function createUrlTableCell(url) {
  const urlCell = document.createElement('td');
  const urlLink = document.createElement('a');
  urlLink.href = url;
  urlLink.innerText = 'LINK';
  urlLink.target = '_blank';
  urlCell.appendChild(urlLink);
  return urlCell;
}

function createTableRow(data) {
  const row = document.createElement('tr');
  row.appendChild(createTableCell(data.headline));
  row.appendChild(createUrlTableCell(data.url));
  row.appendChild(createTableCell(data.sentiment));
  row.appendChild(createTableCell(formatDate(data.date), s=1));
  row.appendChild(createTableCell((data.explanation)));
  row.appendChild(createTableCell((data.entities)));
  return row;
}

function createTable(data) {
  const table = document.getElementById('table');
  table.classList.add('table')
  const headerRow = document.createElement('tr');
  const headers = ['Headline', 'url', 'sentiment', 'Date', 'explanation' , 'entities'];

  headers.forEach(header => headerRow.appendChild(createTableHeader(header)));
  table.appendChild(headerRow);

  data.sort((a, b) => new Date(b.date) - new Date(a.date));

  const rows = data.map(d => {
    const row = createTableRow(d);
    table.appendChild(row);
    return row;
  });

  return rows;
}

function createPlotlyChart(data) {
  const traces = data.reduce((acc, d) => {
    const traceIndex = acc.findIndex(trace => trace.cluster === d.cluster);
    if (traceIndex === -1) {
      acc.push({
        cluster: d.cluster,
        x: [d.x],
        y: [d.y],
        text: [d.headline],
        sentiment: [d.sentiment],
        mode: 'markers',
        marker: { size: 10 },
        hovertemplate: '%{text}<extra></extra>',
      });
    } else {
      acc[traceIndex].x.push(d.x);
      acc[traceIndex].y.push(d.y);
      acc[traceIndex].text.push(d.headline);
      acc[traceIndex].sentiment.push(d.sentiment);
    }
    return acc;
  }, []);

  const layout = {
    title: 'Scatterplot Chart',
    hovermode: 'closest',
    hoverdistance: 10,
    xaxis: { title: 'X Axis', showticklabels: false },
    yaxis: {title: 'Y Axis', showticklabels: false },
    showlegend: false,
    };
    
    const config = { responsive: true };
    const plot = Plotly.newPlot('chart', traces, layout, config);
    return { plot, traces, layout, config };
    }
    
    function updateChart(sentiment, data, traces, layout, config, rows) {
    const filteredTraces = traces.map(trace => {
    const filteredIndices = sentiment
    ? trace.sentiment
    .map((s, i) => (s.toUpperCase() === sentiment.toUpperCase() ? i : -1))
    .filter(i => i !== -1)
    : [...Array(trace.x.length).keys()];
    return {
      ...trace,
      x: filteredIndices.map(i => trace.x[i]),
      y: filteredIndices.map(i => trace.y[i]),
      text: filteredIndices.map(i => trace.text[i]),
      sentiment: filteredIndices.map(i => trace.sentiment[i]),
    };
  });

  Plotly.react('chart', filteredTraces, layout, config);
  
  rows.forEach((row, i) => {
  const x = data[i].x;
  const y = data[i].y;
  const sentimentMatches = !sentiment || data[i].sentiment.toUpperCase() === sentiment.toUpperCase();
  const isInZoomRange = !currentXRange || (x >= currentXRange[0] && x <= currentXRange[1] && y >= currentYRange[0] && y <= currentYRange[1]);

  row.style.display = sentimentMatches && isInZoomRange ? "" : "none";
});

updateScorecards(rows, data.length);
}

// Main part
const rows = createTable(data);
const { plot, traces, layout, config } = createPlotlyChart(data);

plot.then(function () {
const chartDiv = document.getElementById("chart");

chartDiv.on("plotly_relayout", function (eventdata) {
currentXRange = [eventdata["xaxis.range[0]"], eventdata["xaxis.range[1]"]];
currentYRange = [eventdata["yaxis.range[0]"], eventdata["yaxis.range[1]"]];
const resetZoom = eventdata["xaxis.autorange"] && eventdata["yaxis.autorange"];
if (resetZoom) {
  currentXRange = null;
  currentYRange = null;
  resetChartData(traces, layout, config);
}

rows.forEach((row, i) => {
  const x = data[i].x;
  const y = data[i].y;
  const displayRow = resetZoom || (x >= currentXRange[0] && x <= currentXRange[1] && y >= currentYRange[0] && y <= currentYRange[1]);
  row.style.display = displayRow ? "" : "none";
});

updateScorecards(rows, data.length);
});

chartDiv.on("plotly_doubleclick", function () {
currentXRange = null;
currentYRange = null;
resetChartData(traces, layout, config);
rows.forEach(row => {
row.style.display = "";
});
updateScorecards(rows, data.length);
});
});

function resetChartData(traces, layout, config) {
Plotly.react("chart", traces, layout, config);
}

// Add event listeners for the buttons
document.getElementById('show-all').addEventListener('click', () => updateChart(null, data, traces, layout, config, rows));
document.getElementById('show-positive').addEventListener('click', () => updateChart('Positive', data, traces, layout, config, rows));
document.getElementById('show-negative').addEventListener('click', () => updateChart('Negative', data, traces, layout, config, rows));
// Initialize scorecards
updateScorecards(rows, data.length);