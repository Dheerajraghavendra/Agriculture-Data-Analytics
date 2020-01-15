<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="initial-scale=1.0, user-scalable=no" />
<style type="text/css">
  html { height: 100% }
  body { height: 100%; margin: 0px; padding: 0px }
  #map_canvas { height: 100% }
</style>
<script type="text/javascript"
    src="http://maps.google.com/maps/api/js?sensor=false">
</script>
<script type="text/javascript">
/* Load XML file from a GET request used to get the IC XML file */
function loadXMLDoc(XMLname) {
	var xmlDoc;
	if (window.XMLHttpRequest)
		{
		xmlDoc=new window.XMLHttpRequest();
		xmlDoc.open('GET',XMLname,false);
		xmlDoc.send('');
		return xmlDoc.responseXML;
		}
	alert('Error loading document!');
	return null;
}

/* Function used when the user clicks in the marker */
function replaceURL(param)
    {
    	var reportUnit = "/reports/gis/foodstore";
    	var url = JRSPath + "flow.html?_flowId=viewReportFlow&viewAsDashboardFrame=true&reportUnit=" + reportUnit +"&Store_name=" + param; 
		var frame = document.getElementById("report");
       	if(frame == null)
            frame = window.report;
       	frame.src = url;
    }


function setMarkers(map, locations) {

  var image = new google.maps.MarkerImage('images/map_icon.gif',
      new google.maps.Size(31, 28),
      new google.maps.Point(0,0),
      new google.maps.Point(0, 28));
  
  var shape = {
      coord: [1, 1, 1, 28, 31, 28, 31 , 1],
      type: 'poly'
  		};
   
  for (var i = 0; i < locations.length; i++) {
    var store = locations[i];
    var state = store[0];
    var myLatLng = new google.maps.LatLng(store[1], store[2]);
    var marker = new google.maps.Marker({
        position: myLatLng,
        map: map,
        icon: image,
        shape: shape,
        title: store[0],
        zIndex: i+1,
    	});
    createlink(marker,state);
	}
 
}

//Create the links on the Markers

function createlink(marker,paramerter) {
		google.maps.event.addListener(marker, 'click', function() {replaceURL(paramerter);});
}

function initialize() {
  var myOptions = {
    zoom: 5,
    center: new google.maps.LatLng(41.7, -122.4),
    mapTypeId: google.maps.MapTypeId.ROADMAP
  }
  var map = new google.maps.Map(document.getElementById("map_canvas"),myOptions);
  
  var startURL = JRSPath + "flow.html?_flowId=dashboardRuntimeFlow&dashboardResource=" + encodeURI(startReportUnit) + "&viewAsDashboardFrame=true"
  setMarkers(map, stores);

  var frame = document.getElementById("report");
  frame.src = startURL;
}

// Load the GIS info from an IC
// using a Rest call for getting the IC values in XML
// add organizations/organization_1/ for superuser

xmlDoc=loadXMLDoc('rest_v2/reports/reports/gis/storeGIS/inputControls');
var M = xmlDoc.getElementsByTagName('option');
var stores = [];
var JRSPath = '/jasperserver-pro/';
var startReportUnit = '/reports/gis/GIS_START'; // "/supermart/SupermartDashboard30";

for (i=0;i< M.length;i++){
	var value = M[i].getElementsByTagName('value')[0].childNodes[0].nodeValue;
	var gis = M[i].getElementsByTagName('label')[0].childNodes[0].nodeValue;
	if (gis == '[Null]') gis = '0,-16'; // Force Nulls to be in the middle of nowhere
	gis = gis.split(',');
    stores.push([ value, gis[0] , gis[1] ]);
}

</script>

<title>MyJS-GIS Script v0.5</title>
</head>
<body style="margin:0px; padding:0px;" onload="initialize()"> 
<div>
<script type="text/javascript">


</script>
</div>
<div id="map_canvas" style="width: 420px; height: 530px"></div> 
<iframe name="report" id="report" align="right" src="" width="900px" height="530px" style="position:absolute; top:0px; left:422px">  </iframe>
</body>
</html>
