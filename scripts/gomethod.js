
    function selectRandom() {
        $('#group1a').show();
        $('#group2a').hide();
      }
  
      function selectClustered() {
        $('#group1a').hide();
        $('#group2a').show();
      }

  
      var figure = JSON.parse(json);
      var figure2 = JSON.parse(clustered_json);
  
      Plotly.newPlot('method_grapha', figure.data, figure.layout,{responsive: true});
      Plotly.newPlot('method_graph2a', figure2.data, figure2.layout,{responsive: true});