// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
/**
 * @fileoverview Demo which predicts protein functions from amino acid
 * sequence using a TensorFlow.js deep neural network.
 */

/**
 *  {Array.<Array.<Number>>} Outer dimension is the which EC is being predicted.
 *  Inner array dimension is the length of the amino acid sequence.
 *  Values are the color at that residue index. The value is a decimal
 *  representation of the hex color (e.g. 16777215 in hex is 0xFFFFFF).
 **/
var ec_ready;
var go_ready;
let ecPDBColorLists = [];
let correspondingPDB = null;
let schemes = {};
let representations = {};
let pdbView /** {?NGL.Stage} NGL stage for viewing PDB structure. **/ = null;
let /** {boolean} **/ currentlyHasEnzymePrediction = false;

/** {String} PDB id for when the sequence is a custom input sequence. **/
const NO_PDB_STRUCTURE_SENTINEL = 'no_pdb_structure';

var ecNames;

let /** {tf.Model} TensorFlow goModel **/ goModel;
let goParenthood = null;
let goNames = null;
const /** {number} Threshold used to decide whether to make prediction **/GO_DECISION_THRESHOLD = 0.339991183677139;
let /** {Array} Array of EC names keyed by EC numbers **/ vocabList = [];
let vocabLookup = {};

// TODO(theosanderson): set value for final network
const /** {number} **/ KERNEL_SIZE = 91;
let /** {Array} 2D kernel of final layer **/ kernel;
let /** {tf.Model} TensorFlow goModel **/ ecModel;
var b;
/** {Number} Timeout while we're waiting for the user to be done providing input. **/
let waitDoneTypingTimeout = null;
const /** {Array} String amino acid vocabulary **/ AMINO_ACID_VOCABULARY = [
  'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
];
const /** {number} Threshold used to decide whether to make prediction **/EC_DECISION_THRESHOLD = 0.01;

enableCancellablePromises();
let currentLoadingPromise = Promise.resolve();
var ProtVista = require('ProtVista');

load();

tf.disableDeprecationWarnings();

DEFAULT_SEQUENCE_PLACEHOLDER_TEXT = "Type your sequence here";

// http://detectmobilebrowsers.com/
function isInMobileBrowser() {
  var check = false;
  (function (a) {
    if (/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(
        a)
        || /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(
            a.substr(0, 4))) {
      check = true;
    }
  })(navigator.userAgent || navigator.vendor || window.opera);
  return check;
}

function preventScrolling(event) {
  var scrollTo = null;

  if (event.type == 'mousewheel') {
    scrollTo = (event.wheelDelta * -1);
  } else if (event.type == 'DOMMouseScroll') {
    scrollTo = 40 * event.detail;
  }

  if (scrollTo) {
    event.preventDefault();
    $(this).scrollTop(scrollTo + $(this).scrollTop());
  }
}

/**
 * Loads pdb structure with name.
 * @param{string} pdbID.
 * @return{!Promise<void>}
 * */
function loadPDB(pdbID) {
  $('#pdb_unavailable').hide();
  pdbMainContainer = $("#pdb_main_container");
  preventScrollingEl = document.getElementById("pdb_view");
  preventScrollingEl.addEventListener('mousewheel', preventScrolling, {passive: false});
  preventScrollingEl.addEventListener('DOMMouseScroll', preventScrolling, {passive: false});

  if (!pdbView) {
    pdbView = new NGL.Stage("pdb_view", {backgroundColor: 'white'});
  }

  pdbView.removeAllComponents();



  return pdbView.loadFile('rcsb://' + pdbID).then(function (o) {
    drawPredictions();
    pdbView.autoView();
    document.getElementById("pdb_view").onclick = function () {
      pdbView.setSpin(false);
    };
    document.getElementById("pdb_view").onmousedown = function () {
      pdbView.setSpin(false);
    };
    pdbView.setSpin(true);
  }).then(x => {
    pdbMainContainer.show()
  });
}

/**
 * Adds a colored representation to the pdbView for a given residue index.
 * @param{number} i.
 * */
function addRepresentation(i) {
  residuesToColors = ecPDBColorLists[i];
  let scheme = NGL.ColormakerRegistry.addScheme(function (params) {
    this.atomColor = function (atom) {
      return (residuesToColors[atom.residueIndex -
      atom.chainStore.residueOffset[atom.chainIndex]]);
    };
  }, 'scheme_' + i);
  schemes[i] = scheme;
  letter = pdbView.compList[0]
      .structure.chainStore.getChainname(0);
  let rep = pdbView.compList[0].addRepresentation(
      'cartoon', {color: scheme, sele: ':' + letter});

  representations[i] = rep;
}

function highlighter(n) {
  if (correspondingPDB != NO_PDB_STRUCTURE_SENTINEL) {
    for (i in ecPDBColorLists) {
      representations[i].setVisibility(false)
    }
    representations[n].setVisibility(true)
  }
}

function drawPredictions() {
  letter = pdbView.compList[0]
      .structure.chainStore.getChainname(0);
  for (i in ecPDBColorLists) {
    addRepresentation(i)
  }
  pdbView.autoView();
  highlighter(0);
}


/**
 * Returns numeric array after applying moving average
 * @param{!array} inputArray
 * @param{number} kernelSize
 * @return{!array}
 * */
function movingAverage(inputArray, kernelSize) {
  console.assert(kernelSize > 2);
  let oneSide = (kernelSize - 1) / 2;
  let outputArray = [];
  for (let i = 0; i < inputArray.length; i++) {
    let startIndex = Math.max(0, i - oneSide);
    let endIndex = Math.min(inputArray.length, i + oneSide);
    let slice = inputArray.slice(startIndex, endIndex);
    let sum = slice.reduce(function (a, b) {
      return a + b;
    });
    let avg = sum; // / slice.length;
    outputArray.push(avg);
  }
  return (outputArray);
}

/**
 * Return max of numeric array.
 * @param{!array} numericArray
 * @return{!array}
 * */
function getMaxOfArray(numericArray) {
  return Math.max.apply(null, numericArray);
}

/**
 * Sanitize string input, stripping out newlines, whitespace and header lines.
 * @param{string} input
 * @return{string | Error}
 * */
function sanitizeInput(input) {
  let sanitized = input.toUpperCase()
      // Replace first line if it's FASTA format.
      .replace(/>(.*)\n/, '')
      // Get rid of all newlines and spaces.
      .replace(/\s/g, '')
      // Sometimes an asterisk is used as the stop codon; remove that.
      .replace(/\*$/, '');

  if (sanitized.match(/[^\w]|_/)) {
    // There is some punctuation: reject.
    return new Error("Only one sequence (and one header line) are allowed; check for punctuation.")
  }

  return sanitized;
}

/**
 * Create inputs to goModel from string sequence.
 * @param{string} sequence
 * @return{!array} [tf.tensor1d of lengths, tf.tensor2d of one-hot sequence]
 * */
function getInputs(sequence) {
  sequence = sequence.split('');
  let numerical = [];
  for (let char in sequence) {
    numerical.push(AMINO_ACID_VOCABULARY.indexOf(sequence[char]));
  }
  let numericalTensor = tf.tensor1d(numerical, dtype = 'int32');
  let oneHotTensor = tf.oneHot(numericalTensor, AMINO_ACID_VOCABULARY.length);
  let lengthTensor = tf.tensor1d([sequence.length], dtype = 'int32');

  oneHotTensor = oneHotTensor.expandDims();
  oneHotTensor = tf.cast(oneHotTensor, 'float32');
  return [lengthTensor, oneHotTensor];
}

function parseECVocab(v) {
  vocabList = []
  vocabLookup = {}
  let descs = v.split('\n');
  for (let d in descs) {
    let splitUp = descs[d].split('\t');
    if (splitUp[0] != "vocab_item") {
      // This creates a lookup from a string EC number to a string description.
      vocabList.push(splitUp[0])
      vocabLookup[splitUp[0]] = vocabList.length - 1
    }
  }
}

/**
 * Loads metadata associated with vocabulary.
 */
function loadECMetadata() {
  let ecVocabPromise = $.ajax({
                url: './vocabs/EC.tsv',
                type: "GET",
                dataType: "text",
                mimeType: "text/plain",
                success: parseECVocab});
  let ecNamesPromise = $.getJSON("./ec_names.json", function (data) {
    ecNames = data;
  });
  return Promise.all([ecVocabPromise, ecNamesPromise]);
}

/**
 * Perform inference with ecModel, and limited post-processing of results.
 * @param{string} input_sequence
 * @return{!array} [values, residue_info_for_top_k]
 */
async function performECInference(input_sequence) {
  
  if (isInMobileBrowser()) {
    throw Error("Not supported in mobile browser.")
  }
  console.log("Starting EC inference");
  let results = ecModel.execute(
      getInputs(input_sequence),
      ['Sigmoid', 'set_padding_to_sentinel_2/Select']);
  let overallProbabilities = results[0];
  let topk = overallProbabilities.squeeze().topk(40);
  let representation = results[1];
  let values = await topk.values.array();
  let indices = await topk.indices.array();
  let boolIncluded = await
      tf.greater(overallProbabilities, tf.scalar(EC_DECISION_THRESHOLD)).array();
  boolIncluded = boolIncluded[0];

  //Now lets remove nodes that aren't at the edges.

  for (i in boolIncluded) {
    if (boolIncluded[i]) {
      parts = vocabList[i].split(".")

      for (var partNumber=1 ; partNumber< parts.length; partNumber++) {
        if (partNumber == parts.length - 1 | parts[partNumber + 1] != "-") {
          
          parentalEC = parts.slice(0, partNumber ).concat(Array(parts.length - partNumber ).fill("-"))
          parentalEC = parentalEC.join(".")
          
          boolIncluded[vocabLookup[parentalEC]] = 0
        }
      }

    }

  }



  for (i in indices) {
    if (!boolIncluded[indices[i]]) {
      values[i] = 0
    }
  }
  b = values
  boolIncluded = tf.tensor1d(boolIncluded).expandDims(0);

  let total =
      tf.cast(boolIncluded, "int32").sum();

  let classByPosition = await tf.matMul(representation.squeeze(), kernel);
  if (total > 1 & 0) { //never run
    // We set non-positive classes activations to zero, then clip only to
    // positive activations, and sum across all classes:
    let classByPositionReduced = classByPosition.mul(boolIncluded)
        .clipByValue(0, 100000000)
        .sum(1, keepDims = true);
    // Now we multiply the original array by 2, then subtract this sum. This
    // means that a => 2a - (a+b+c+d) (i.e. scale of a is preserved):
    classByPosition =
        await classByPosition.mul(total).sub(classByPositionReduced);
  }

  let residueInfoForTopk =
      await classByPosition.gather(topk.indices, axis = 1).transpose().array();
  return ([values, indices, residueInfoForTopk]);
}

/**
 * Generate HTML output for one class prediction.
 * A value of 1 is completely green (#00ff00), which fades to gray until an
 * input value of .5 (#bfbfbf), which fades to white until a value of 0.
 * @param{!Number} float between 0 and 1.
 * @return{!Array} rgb array of 3 ints.
 */
function get_color_from_value(value) {
  if (value < 0.5) {
    proportion = value * 2;
    start = [255, 255, 255];
    end = [180, 180, 180]
  } else {
    proportion = (value - 0.5) * 2;
    start = [180, 180, 180];
    end = [0, 255, 0]
  }
  out = [Math.round(start[0] * (1 - proportion) + end[0] * proportion),
    Math.round(start[1] * (1 - proportion) + end[1] * proportion),
    Math.round(start[2] * (1 - proportion) + end[2] * proportion)];
  return (out)
}

function getSingleOutput(vocabIndex, overallValue, values, sequence) {
  let start_val = 0;
  let start_pt = 0;
  let features = [];
  representationColorList = [];
  values = movingAverage(values, KERNEL_SIZE);
  let maxCValue = getMaxOfArray(values);
  fname =
      "<div class='large_pdb_controls_entry_contents'>"
        + "<a href='https://enzyme.expasy.org/EC/" + vocabList[vocabIndex].split(":")[1] + "' "
          + "class='top-figure-link' "
          + "target='_blank'>"
          + vocabList[vocabIndex] +
        "</a>: " +
      ecNames[vocabList[vocabIndex].replace("EC:", "")] +
      "</div>";
  for (cIndex in sequence) {
    let theValue = values[cIndex];

    theValue = Math.max(theValue, 0) / maxCValue;
    diff = Math.abs(theValue - start_val);

    let color = get_color_from_value(theValue);

    if (diff > 0.01 || cIndex == sequence.length - 1) {

      features.push(make_feature(fname, color, start_pt, cIndex));
      start_pt = parseInt(cIndex) + 1;
      start_val = theValue
    }
    representationColorList.push(getDecimalColor(color))
  }

  return [features, representationColorList, fname];
}

var rgbToHex = function (rgb) {
  var hex = Number(rgb).toString(16);
  if (hex.length < 2) {
    hex = "0" + hex;
  }
  return hex;

};

/**
 * Converts an [R,G,B] color to an integer hex value.
 * @param{!array} color
 * @return{number}
 */
function getDecimalColor(color) {
  let dec = Math.round(
      Math.round(color[0]) * 256 * 256 + Math.round(color[1]) * 256 +
      Math.round(color[2]));
  return dec;
}

var fullColorHex = function (r, g, b) {
  var red = rgbToHex(r);
  var green = rgbToHex(g);
  var blue = rgbToHex(b);
  return "#" + red + green + blue;
};

function make_feature(the_name, color, position_start, position_end) {
  hexcolor = fullColorHex(color[0], color[1], color[2]);
  feature = {
    "type": the_name,
    "category": "Predictions",
    "begin": position_start,
    "end": position_end,
    //Color used to distinguish from default data sources features

    "color": hexcolor
  };
  return (feature)
}

/**
 *  Generate HTML output from predictions.
 * @param{!array} topkScores The scores for the top k predictions
 * @param{!array} residueInfoForTopk Contributions from residues for the top 10
 * predictions
 * @param{!float} threshold Threshold for positive predictions
 * @param{!array} indices Indices in vocab
 * @param{string} input Input sequence
 * */
featureNames = [];

function generateECOutput(
    topkScores, residueInfoForTopk, threshold, indices, input) {
  allFeatures = [];
  featureNames = [];
  predictedSomething = false;
  ecPDBColorLists = [];
  for (var i in topkScores) {
    if (topkScores[i] > threshold) {
      predictedSomething = true;
      let overallValue = topkScores[i];
      let sequence = input.split('');
      let theseValues = residueInfoForTopk[i];
      [features, representationColorList, fname] = getSingleOutput(indices[i], overallValue,
          theseValues, sequence);
      allFeatures.push(...features);
      ecPDBColorLists.push(representationColorList);
      featureNames.push(fname)
    }
  }
  currentlyHasEnzymePrediction = predictedSomething;
  if (predictedSomething) {
    htmlOption = "";
    for (fn in featureNames) {
      featureName = featureNames[fn];
      if (fn == 0) {
        selected = "checked"
      } else {
        selected = ""
      }
      htmlOption += "<div class='large_pdb_controls_entry'>"
          + "<input "
          + "type='radio' " + selected + " "
          + "id='rad" + fn + "' "
          + "name='repr' "
          + "class='radiolabel' "
          + "onclick='highlighter(" + fn + ")' />"
          + "<label for='rad" + fn + "'>" + featureName + "</label>"
          + "</div>"
    }
    $('#activation_options').html(htmlOption);
    var yourDiv = document.getElementById('protvista');
    new ProtVista({
      el: yourDiv,

      //This will be **always** added at the end of your data source URL but before the extension
      uniprotacc: 'P05067',

      //Default sources will be included (even if this option is omitted)
      defaultSources: false,

      //Your data sources are defined here
      customDataSource: {
        url: './data/externalFeatures_',
        source: 'myLab',
        useExtension: true,
        stuff: {
          "sequence": input,
          "features": allFeatures,
        }
      }
    })
  }
  $('#protvista .up_pftv_track-header').each(function (index) {
    $(this).mouseover(function () {
      highlighter(index)
    });
  });
  $('#protvista .up_pftv_track').each(function (index) {
    $(this).mouseover(function () {
      highlighter(index)
    });
  });
  setTrackHighlightOpacityHandlers();
  setInitialTrackHighlightToFirst();
  removeHyperLinkForProtvistaCredits();


  // The titles generated by protvista lead to messy tooltips,
  // so we remove them.
  $(".up_pftv_track-header").removeAttr("title");

  // Add axis label to residue number.
  $("#ec_container>div>div>div>.up_pftv_aaviewer").append(
      $("<div style='text-align: center;'></div>").text("Amino acid index"))
}

function setInitialTrackHighlightToFirst() {
  $(".up_pftv_track-header:first")
      .css('font-weight', 500)
      .next() // The next element is the track.
      .css('opacity', 1);
}

function resetTrackHighlightOpacity() {
  // Unset highlighting on all tracks.
  $('#protvista .up_pftv_track-header')
      .css('font-weight', 'unset')
      .next() // The next element is the track.
      .css('opacity', .5);
}

/**
 * If the user hovers over the track or track header, increase the opacity/
 * font-weight of that track.
 */
function setTrackHighlightOpacityHandlers() {
  // Set for track header.
  $('#protvista .up_pftv_track-header')
      .mouseover(x => resetTrackHighlightOpacity())
      .mouseover(x => $(x.currentTarget).css('font-weight', 500))

      // If the track header is highlighted, get the next element, which is
      // the actual track, and then set the opacity.
      .mouseover(x=>$(x.currentTarget).next().css('opacity', 1));

  // Set for track.
  $('#protvista .up_pftv_track')
      .mouseover(x => resetTrackHighlightOpacity())
      .mouseover(x => $(x.currentTarget).css('opacity', 1))

      // If the track is highlighted, get the previous element, which is
      // the track header, and then set the font weight.
      .mouseover(x => $(x.currentTarget).prev().css('font-weight', 500));
}


/**
 *  Make prediction and generate output.
 * */
async function makeECPrediction(sanitizedInput) {
  let [values, indices, residueInfoForTopk] = await performECInference(sanitizedInput);

  generateECOutput(
      values, residueInfoForTopk, EC_DECISION_THRESHOLD, indices, sanitizedInput);
  $("#ec_loader").hide();
  $("#enzyme_stuff").show();
  switchToAppropriateViewMode();
  $("#go_content_container").hide();
  $("#go_content").show();
  $("#go_loader").show();
  setTimeout(x=>makeGOPrediction(sanitizedInput),1);
  console.log("completed EC inference")

}

function hideLoaders() {
  $("#ec_loader").hide();
  $("#go_loader").hide();
}

function hideContent() {
  $("#enzyme_stuff").hide();
  $("#go_container").hide();
}

function hideContentShowLoaders() {
  $("#protvista").hide();
  $("#pdb_main_container").hide();

  $("#go_content_container").hide();

  // Need to show containers of loaders to show loaders.
  $("#enzyme_stuff").show();
  $("#go_container").show();

  $("#ec_loader").show();
  $("#go_loader").show();
}

function hideContentShowError(err) {
  $("#input_seq_error_cont").show();
  $("#input_seq_error_cont").html(err);
  hideContent();
}

function waitTilDoneTypingThenMakePrediction(event) {
  if (event.keyCode>=9 && event.keyCode <= 45) {
    return; // Non-input keycodes or whitespace.
  }
  hideContentShowLoaders();
  clearTimeout(waitDoneTypingTimeout);
  waitDoneTypingTimeout = setTimeout(function () {
    customInputSeq()
  }, 500);
}

/**
 * Predicts using EC and GO models for the text box's input_seq value, then
 * switches to the appropriate View mode (e.g. stays in GO view mode if coming
 * from that mode).
 *
 * If the input_seq is invalid, updates DOM to indicate that error, and returns
 * a fulfilled Promise.
 *
 * @return {Promise<void>}
 * */
function startPrediction() {
  console.log("start prediction called")

  inputSequence = document.getElementById('input_seq').value;
  if (! inputSequence) {
    hideLoaders();
    hideContent();
    return Promise.resolve();
  }
  let inputString = sanitizeInput(inputSequence);

  if (inputString instanceof Error) {
    hideContentShowError(inputString.toString());
    return Promise.resolve();
  }

  if (inputString.length <= 40) {
    hideContentShowError('Please enter a sequence of more than 40 amino acids for prediction.');
    return Promise.resolve();
  }

  $("#input_seq_error_cont").hide();
  gtag('event', 'input_sequence', {'event_category': 'infer'});

  makeECPrediction(inputString);

  
}


function loadPrecomputedExample(filename) {
  jqXHRPromise = $.getJSON(filename, function (data) {
    topkScores = data['EC']['topkScores'];
    residueInfoForTopk = data['EC']['residueInfoForTopk'];
    threshold = data['EC']['threshold'];
    indices = data['EC']['indices'];
    input = data['EC']['input'];
    generateECOutput(topkScores, residueInfoForTopk, threshold, indices, input);

    topkScores = data['GO']['topkScores'];
    threshold = data['GO']['threshold'];
    indices = data['GO']['indices'];
    generateGOOutput(topkScores, threshold, indices)
  });

  // https://stackoverflow.com/questions/24315180/how-to-cast-jquery-ajax-calls-to-bluebird-promises-without-the-deferred-anit-p
  vanillaPromise = Promise.resolve(jqXHRPromise);
  return vanillaPromise;
}

function showHideForCustomInputSeq() {
  if (pdbViewIsLarge()) {
    switchToSummaryView(); // No PDB structures are available for custom sequences.
  }

  $('.eg_selected').removeClass('eg_selected');
  $('#yourseq').addClass('eg_selected');
  $('#protein_description_container').hide();

  hideContentShowLoaders();
}

function waitFor(selector) {
  return new Promise(function (res, rej) {
    waitForElementToDisplay(selector, 200);
    function waitForElementToDisplay(selector, time) {
      if ($(selector).length) {
        res($(selector));
      }
      else {
        setTimeout(function () {
          waitForElementToDisplay(selector, time);
        }, time);
      }
    }
  });
}

/**
 * Determines whether the pdb view is large for use in choosing the correct
 * view (e.g. summary, PDB or GO view).
 * @return {boolean}
 */
function pdbViewIsLarge() {
 return $("#pdb_view").hasClass("pdb_large_viewer");
}

/**
 * Determines whether the go view is large for use in choosing the correct
 * view (e.g. summary, PDB or GO view).
 * @return {boolean}
 */
function goViewIsLarge() {
  return $("#cy").hasClass("cy_large_viewer");
}

/**
 * Given whether the go/pdb views are large, switch to the appropriate view.
 */
function switchToAppropriateViewMode() {
  if (pdbViewIsLarge()) {
    switchToPDBView();
  }
  if (goViewIsLarge()) {
    switchToGOView();
  }

  if (!pdbViewIsLarge() && !goViewIsLarge()) {
    switchToSummaryView()
  }
}

function clearInputSeq() {
  $("#input_seq").val("");
}

function customInputSeq() {
  ec_ready=false;
  go_ready=false;
  correspondingPDB = NO_PDB_STRUCTURE_SENTINEL;

  showHideForCustomInputSeq();
  if (isInMobileBrowser()) {
    hideContentShowError("Inference for custom sequences isn't supported in mobile browsers.")
  } else {
    loadDemo(null, null, null, null).then(x=>hideContentShowLoaders())
        .then(completed => startPrediction())

        // Wait until predictions are done to explain the PDB missing.
        // If it shows up before there are predictions, it looks pretty ugly.
        .then(completed => $('#pdb_unavailable').show());
  }
}

/**
 * For use with bluebird promises.
 * Apparently needs to be called before each call to cancel if the promise
 * was created since configuration.
 */
function enableCancellablePromises() {
  Promise.config({cancellation: true,});
}

/**
 * @param inputSeqContents {string} Contents for text area. If null, no text is
 *        set.
 * @param proteinDescription {string} Description of text area contents.
 * @param proteinMoreInfoLink {string} link e.g. to uniprot.
 * @param pdb {string} pdb identifier for 3d structure. Allowed
 *        to be falsy if no PDB structure is known.
 * @param precomputedExamplePath {string} path to JSON file on server. Allowed
 *        to be falsy if no precomputed example is desired.
 * @return {Promise<void>} on completion.
 */
function loadDemo(inputSeqContents, proteinDescription, proteinMoreInfoLink,
    pdb, precomputedExamplePath) {
  if (pdb == correspondingPDB) { // Reclicked on same example.
    // User probably expects to return to summary mode.
    return Promise.resolve(switchToSummaryView());
  }
  enableCancellablePromises();
  currentLoadingPromise.cancel();

  hideContentShowLoaders();

  // Use waitFor because on first load, the elements mightn't be there yet.
  let inputSeqPromise = inputSeqContents
      ? waitFor("#input_seq").then(x => x.val(inputSeqContents))
      : Promise.resolve();
  let proteinDescContentsPromise = waitFor("#protein_description_contents")
      .then(x => x.html(proteinDescription));
  let showProteinDescPromise = proteinDescription
      ? waitFor("#protein_description_container").then(x => x.show())
      : Promise.resolve();
  let proteinInfoLinkPromise = waitFor("#protein_more_info_link")
      .then(x => x.attr('href', proteinMoreInfoLink))
      .then(x => x.show());
  let hideUnavailablePDBPromise = waitFor("#pdb_unavailable").then(x=>x.hide());

  let precomputedExamplePromise = precomputedExamplePath
      ? loadPrecomputedExample(precomputedExamplePath)
      : Promise.resolve();

  let inputErrorPromise = waitFor("#input_seq_error_cont").then(x=>x.hide());

  currentLoadingPromise = Promise.all([
    inputSeqPromise,
    proteinDescContentsPromise,
    showProteinDescPromise,
    proteinInfoLinkPromise,
    hideUnavailablePDBPromise,
    precomputedExamplePromise,
    inputErrorPromise,
  ]);

  if (pdb) {
    correspondingPDB = pdb;
    currentLoadingPromise = currentLoadingPromise.then(x=>loadPDB(pdb))
  } else {
    correspondingPDB = NO_PDB_STRUCTURE_SENTINEL;
  }

  currentLoadingPromise = currentLoadingPromise.then(x=>switchToAppropriateViewMode());

  return currentLoadingPromise;
}


function markExampleAsHighlighted(event) {
  $('.eg_selected').removeClass('eg_selected');
  $('.arrow_box').removeClass('arrow_box');

  $(event.currentTarget).addClass('eg_selected');
  $(event.currentTarget).addClass('arrow_box');
}

function loadDemoHeme() {
  inputSeqContents = `>sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo sapiens OX=9606 GN=HBA1 PE=1 SV=2
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTP
AVHASLDKFLASVSTVLTSKYR`;
  proteinDescription = "<b>Nonenzyme.</b> Involved in oxygen transport.";
  proteinMoreInfoLink = "https://www.uniprot.org/uniprot/P69905";
  loadDemo(inputSeqContents, proteinDescription, proteinMoreInfoLink, null,'precomputed/homo_sapiens_hemoglobin.json');
}

function loadAmylase() {
  inputSeqContents = `>sp|P04745|AMY1_HUMAN Alpha-amylase 1 OS=Homo sapiens OX=9606 GN=AMY1A PE=1 SV=2
MKLFWLLFTIGFCWAQYSSNTQQGRTSIVHLFEWRWVDIALECERYLAPKGFGGVQVSPPNENVAIHNPFRPWWERYQPVSYKLCTRSGNEDEFRNMVTRCNNVGVRIYVDAVINHMCGN
AVSAGTSSTCGSYFNPGSRDFPAVPYSGWDFNDGKCKTGSGDIENYNDATQVRDCRLSGLLDLALGKDYVRSKIAEYMNHLIDIGVAGFRIDASKHMWPGDIKAILDKLHNLNSNWFPEG
SKPFIYQEVIDLGGEPIKSSDYFGNGRVTEFKYGAKLGTVIRKWNGEKMSYLKNWGEGWGFMPSDRALVFVDNHDNQRGHGAGGASILTFWDARLYKMAVGFMLAHPYGFTRVMSSYRWP
RYFENGKDVNDWVGPPNDNGVTKEVTINPDTTCGNDWVCEHRWRQIRNMVNFRNVVDGQPFTNWYDNGSNQVAFGRGNRGFIVFNNDDWTFSLTLQTGLPAGTYCDVISGDKINGNCTGI
KIYVSDDGKAHFSISNSAEDPFIAIHAESKL`;
  proteinDescription = "<b>Single-function enzyme</b>. Found chiefly in saliva and pancreatic fluid, that converts starch and glycogen into simple sugars.";
  proteinMoreInfoLink = "https://www.uniprot.org/uniprot/P04745";
  loadDemo(inputSeqContents, proteinDescription, proteinMoreInfoLink, '1C8Q','precomputed/homo_sapiens_amylase.json');
}

function loadDemoEColi() {
  inputSeqContents = `>sp|P00909|TRPC_ECOLI Tryptophan biosynthesis protein TrpCF OS=Escherichia coli (strain K12) OX=83333 GN=trpC PE=1 SV=4
MMQTVLAKIVADKAIWVEARKQQQPLASFQNEVQPSTRHFYDALQGARTAFILECKKASPSKGVIRDDFDPARIAAIYKHYASAISVLTDEKYFQGSFNFLPIVSQIAPQPILCKDFIID
PYQIYLARYYQADACLLMLSVLDDDQYRQLAAVAHSLEMGVLTEVSNEEEQERAIALGAKVVGINNRDLRDLSIDLNRTRELAPKLGHNVTVISESGINTYAQVRELSHFANGFLIGSAL
MAHDDLHAAVRRVLLGENKVCGLTRGQDAKAAYDAGAIYGGLIFVATSPRCVNVEQAQEVMAAAPLQYVGVFRNHDIADVVDKAKVLSLAAVQLHGNEEQLYIDTLREALPAHVAIWKAL
SVGETLPAREFQHVDKYVLDNGQGGSGQRFDWSLLNGQSLGNVLLAGGLGADNCVEAAQTGCAGLDFNSAVESQPGIKDARLLASVFQTLRAY`;
  proteinDescription = "<b>Bifunctional enzyme.</b> Catalyzes steps of tryptophan biosynthetic pathway: an isomerase and a synthase.";
  proteinMoreInfoLink = "https://www.uniprot.org/uniprot/P00909";
  loadDemo(inputSeqContents, proteinDescription, proteinMoreInfoLink, '1PII','precomputed/ecoli_trpcf.json');
}

/** Avoid having users click out to help page, but point them via tooltip. */
function removeHyperLinkForProtvistaCredits() {
  protvistaHelpPageLink = $(".up_pftv_credit_container > a");
  protvistaHelpPageLink.attr('title', protvistaHelpPageLink.attr('href'));
  protvistaHelpPageLink.removeAttr('href');
}

function maybeShowMobileBrowserWarning() {
  if (isInMobileBrowser()) {
    waitFor("#use_desktop_browser").then(x => x.show());
  }
}

function hideContentIfInToolOnlyMode() {
  let searchParams = new URLSearchParams(window.location.search);
  let toolOnlyMode = searchParams.has('toolOnly');
  if (toolOnlyMode) {
    waitFor("d-article").then(x => x.hide());
  }
}

/**
 * Load the goModel.
 * */
async function loadGOModel() {
  let cloud_storage_path = 
     'https://storage.googleapis.com/brain-genomics-public/research/proteins/tmp/js_smmdl2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-13901080.3/model.json';

  goModel = await tf.loadGraphModel(
      cloud_storage_path,
      {credentials: 'same-origin'});

  $('#loading_model').hide();
 // $('#loading').hide();
}

async function loadECModel() {
  let cloud_storage_path = 
     'https://storage.googleapis.com/brain-genomics-public/research/proteins/tmp/js_smmdl2_cnn_swissprot_ec_random_swiss-cnn_for_swissprot_ec_random-13897742.6/model.json';
  ecModel = await tf.loadGraphModel(
      cloud_storage_path,
      {credentials: 'same-origin'});

  kernel = await ecModel.execute(getInputs('AAA'), 'logits/kernel');
}

/**
 * Load the ecModel.
 * */
async function load() {
  hideContentIfInToolOnlyMode();
  maybeShowMobileBrowserWarning();
  ecPromise = loadECMetadata();
  goPromise = loadGOMetadata();
  Promise.all([ecPromise, goPromise])
      .then(x=> loadDemoEColi())
      .then(x=>{
          bindCanvasHover();

          // Show magnifying glass text only after demo has been loaded.
          $('.magnifying_glass_text').show();
      });
  if (!isInMobileBrowser()) {
    Promise.all([loadECModel(), loadGOModel()]).then(x=>{
      $('#input_seq').prop("disabled", false);
      $("#input_seq").css("background-image", "url(edit-icon.svg)");
    })
  }

  $.ready(x=>{
    configureTooltips();
  });
}

function configureTooltips() {
  $(document).tooltip({
    show: {
      delay: 0
    },tooltipClass: "tooltip",
  });
}


/**
 *  Generate HTML output from predictions.
 * @param{!array} topkScores The scores for the top k predictions
 * @param{!float} threshold Threshold for positive predictions
 * @param{!array} indices Indices in vocab
 * @param{string} input Input sequence
 * */
function generateGOOutput(
    topkScores, threshold, indices) {
  let itemsForGraph = {};

  topKFiltered = topkScores.filter(x => x > threshold);

  // Compute parent nodes, and include them even if they're below the threshold.
  allLabels = [];
  addToGraph = true;

  for (var i in topKFiltered) {
    let label = goVocab[indices[i]];
    allLabels.push(label)
  }
  try{
  parentNodesToInclude = missingParentNodes(allLabels);
  }
  catch(err){
    addToGraph = false;

  }
  if(addToGraph){

  for (var i in topkScores) {
    if (topkScores[i] > threshold || parentNodesToInclude.includes(
        goVocab[indices[i]])) {
      let overallValue = topkScores[i];
      let label = goVocab[indices[i]];
      itemsForGraph[label] = overallValue;
    }
  }
}

  drawTopPreds(itemsForGraph);

  drawGraph(itemsForGraph);
}

function drawTopPreds(itemsForGraph) {
  goParenthoodFastLookup = {}
  for (const k in goParenthood) {
    goParenthoodFastLookup[k] = new Set(goParenthood[k]);
  }
  candidates = Object.keys(itemsForGraph).filter(
      x => isALeafNode(x, Object.keys(itemsForGraph), goParenthoodFastLookup));
  sortedCandidates = candidates.sort(
      (x, y) => itemsForGraph[y] - itemsForGraph[x]);
  toPrint = sortedCandidates.filter(
      x => itemsForGraph[x] > GO_DECISION_THRESHOLD);
  htmlToSet = "";
  for (var i in toPrint) {
    htmlToSet += "<div>"
                 + "<a href=http://amigo.geneontology.org/amigo/term/" + toPrint[i] + " "
                   + "class='top-figure-link' "
                   + "target='_blank'>"
                   + toPrint[i]
                 + "</a>" + ": "
                 + goNames[toPrint[i]]
              + " </div> ";
  }

  $("#go_grid").html(htmlToSet);
}

function missingParentNodes(listOfNodes) {
  toReturn = listOfNodes.slice();
  for (let i = 0; i < toReturn.length; i++) {
    if (listOfNodes.length>200){
      console.log("Too many GO predictions, unsafe input?")
      throw Error("Error in GO rendering")
    }
    node = toReturn[i];
    parents = goParenthood[node];
    for (p in parents) {
      parent = parents[p];
      if (toReturn.indexOf(parent) == -1) {
        toReturn.push(parent);
      }
    }
  }

  // Get rid of initial child nodes.
  return toReturn.slice(listOfNodes.length)
}

function isALeafNode(n, onlyConsiderTheseNodes, goParenthoodFastLookup) {
  onlyConsiderTheseNodes = new Set(onlyConsiderTheseNodes);
  for (var i in goParenthood) {
    if (onlyConsiderTheseNodes.has(i) && goParenthoodFastLookup[i].has(n)) {  // n has a parent in the subgraph.
      return false;
    }
  }
  return true;
}

var g;
var nodes;

function setCySiblingHighlight() {
  cy.on('mouseover', 'node', function(e) {
    var sel = e.target;
    cy.elements()
        .difference(sel.outgoers()
            .union(sel.incomers()))
        .not(sel)
        .addClass('semitransp');
    sel.addClass('highlight')
        .outgoers()
        .union(sel.incomers())
        .addClass('highlight');
  });
  cy.on('mouseout', 'node', function(e) {
    var sel = e.target;
    cy.elements()
        .removeClass('semitransp');
    sel.removeClass('highlight')
        .outgoers()
        .union(sel.incomers())
        .removeClass('highlight');
  });
}

function drawGraph(items) {
  nodes = Object.keys(items);

  nodeElements = [];

  // Automatically label each of the nodes
  nodes.forEach(function (x) {
    item = {
      data: {
        label: goNames[x] + "\n",
        id: x,
        value: items[x],
        labelvalue: goNames[x] + " (" + items[x].toFixed(2) + ")"
      }
    };
    nodeElements.push(item);
  });

  nodes.forEach(function (x) {
    if (x in goParenthood) {
      parents = goParenthood[x].filter(value => goVocab.includes(value));
      parents.forEach(function (y) {
        item = {data: {source: y, target: x}};
        nodeElements.push(item);
      });
    }
  });

  window.cy = cytoscape({
    container: document.getElementById('cy'),
    style: baseCytoscapeStylesheet(),
    layout: {
      name: 'dagre',
      nodeDimensionsIncludeLabels: true
    },
    autoungrabify: true,

    elements: nodeElements,
    
    wheelSensitivity: .3,
  });

  // Make it so that clicking and dragging on elements pans.
  cy.$("*").panify();

  setCySiblingHighlight();
  setCyPanzoom();
  cy.fit();
}

function baseCytoscapeStylesheet() {
  return cytoscape.stylesheet()
      .selector('node')
      .css({
        'content': 'data(label)',
        'color': 'black',
        'width': 'label',
        'height': 'label',
        'spacingFactor': 0.1,
        'text-valign': 'center',
        'outline': 'none',
        'shape': 'round-rectangle',
        'text-wrap': 'wrap',
        'text-max-width': 30,
        'min-zoomed-font-size': 0,
        'background-color': 'mapData(value, ' + GO_DECISION_THRESHOLD
            + ', 1, #888, #ffdf78)',
        'padding': '5px',
        'text-margin-y': '-8px',
      })
      .selector('edge')
      .css({
        'line-color': 'lightgray',
        'arrow-scale': '2',
        'curve-style': 'taxi',
        'taxi-direction': 'downward',
        'target-arrow-shape': 'triangle'
      })
      .selector('node, edge')
      .css({
        'overlay-opacity': 0, // Disable click animation.
      });
}

function turnOnHoverActionsForCytoscape() {
  cy.style(cy.style()
      .selector('node.highlight')
      .css({
        'border-color': '#000',
        'border-width': '1px',
        'content': 'data(labelvalue)',
      })
      .selector('node.semitransp')
      .css({'opacity': '0.1'})
      .selector('edge.highlight')
      .css({'mid-target-arrow-color': '#FFF'})
      .selector('edge.semitransp')
      .css({'opacity': '0.2'})
      .update()
  );
}

function turnOffHoverActionsForCytoscape() {
  cy.style(baseCytoscapeStylesheet());
}

function hideCyPanzoom() {
  $(".cy-panzoom").hide();
}

function showCyPanzoom() {
  $(".cy-panzoom").show();
}

function setCyPanzoom() {
  cy.panzoom({zoomOnly: true});
}

/**
 *  Make prediction and generate output.
 *  */
async function makeGOPrediction(sanetizedInput) {
  if (!goModel && !isInMobileBrowser()) {
    await loadGOModel();
  }

  let [values, indices] = await performGOInference(sanetizedInput);
  generateGOOutput(
      values, GO_DECISION_THRESHOLD, indices);
 // $('#loading').hide();
  $("#go_content_container").show();
  $("#go_content").hide();
  $("#go_loader").hide();
 switchToAppropriateViewMode();
}

function loadGOMetadata() {
  parenthoodPromise = $.getJSON('models/go/go_parenthood.json', function (data) {
    goParenthood = data;
  });
  namesPromise = $.getJSON('models/go/go_names.json', function (data) {
    goNames = data;
  });

  vocabPromise = $.getJSON("models/go/go_vocab.json", function (data) {
    goVocab = data
  });
  return Promise.all([parenthoodPromise, namesPromise, vocabPromise]);
}

/**
 * Create inputs to goModel from string sequence.
 * @param{string} sequence
 * @return{!array} [tf.tensor1d of lengths, tf.tensor2d of one-hot sequence]
 * */
function getInputs(sequence) {
  sequence = sequence.split('');
  let numerical = [];
  for (let char in sequence) {
    numerical.push(AMINO_ACID_VOCABULARY.indexOf(sequence[char]));
  }
  let numericalTensor = tf.tensor1d(numerical, dtype = 'int32');
  let oneHotTensor = tf.oneHot(numericalTensor, AMINO_ACID_VOCABULARY.length);
  let lengthTensor = tf.tensor1d([sequence.length], dtype = 'int32');

  oneHotTensor = oneHotTensor.expandDims();
  oneHotTensor = tf.cast(oneHotTensor, 'float32');
  return {
    "sequence_length_placeholder": lengthTensor,
    "batched_one_hot_sequences_placeholder": oneHotTensor
  };
}

/**
 * Perform inference with goModel, and limited post-processing of results.
 * @param{string} input_sequence
 * @return{!array} [values, residue_info_for_top_k]
 */
// TODO(mlbileschi): update documentation for new return type.
async function performGOInference(input_sequence) {
  // TODO(theosanderson): second argument here will be a constant in new models
  if (isInMobileBrowser()) {
    throw Error("Not supported in mobile browser.")
  }

  let overallProbabilities = goModel.execute(
      getInputs(input_sequence),
      ['Sigmoid']);
  let topk = overallProbabilities.squeeze().topk(32102);
  let values = await topk.values.array();
  let indices = await topk.indices.array();
  return ([values, indices]);
}

function rerenderPDBAndGO() {
  pdbView.handleResize();
  pdbView.setSpin(true);

  cy.resize();
  cy.fit();
}

/**
 * Adds click handlers to pdb viewer and cytoscape viewport for entering
 * large PDB viewing mode and large GO viewing mode.
 */
function turnOnClickToEnterLargeModes() {
  // Prevent multiple identical handlers from accumulating.
  $("#pdb_view").off('click');
  $("#cy").off('click');

  // Reregister ability to switch to PDB and GO when going to non-zoomed-in mode.
  $("#pdb_view").click(x=>switchToPDBView());
  $("#cy").click(x=>switchToGOView());
}


function makePDBSmall() {
  $('#pdb_view').removeClass('pdb_large_viewer');
  $('#pdb_view').addClass('pdb_small_viewer');
}

function makeGOSmall() {
   $('#cy').removeClass('cy_large_viewer');
   $('#cy').addClass('cy_small_viewer');
}

function makePDBAndGOSmall() {
  makePDBSmall();
  makeGOSmall();
}

function disableInteractionsForSummaryView() {
  cy.zoomingEnabled(false);
  pdbView.mouseControls.remove('scroll');
  pdbView.mouseControls.remove('drag');
}

function disableNGLTooltip() {
  pdbView.mouseControls.remove('hoverPick');
}

function enableNGLTooltip() {
  pdbView.mouseControls.add("hoverPick", NGL.MouseActions.tooltipPick);
}

function enableInteractions() {
  cy.zoomingEnabled(true);
  pdbView.mouseControls.add("scroll", NGL.MouseActions.zoomScroll);

  // Mobile interactions.
  pdbView.mouseControls.add("drag", NGL.MouseActions.rotateDrag);
}

function toggleHelpButtonText(button) {
  if (button.text().match(/more/)) {
    button.text("Less");
  } else {
    button.text("Learn more");
  }
}

function toggleSummaryHelpText(button, contents) {
  if (button.hasClass("help_visible")) {
    button.animate({ width: "88px"}, 'slow', x=>toggleHelpButtonText(button));
    contents.slideUp();
  } else {
    button.animate({ width: "100%"}, 'slow', x=>toggleHelpButtonText(button));
    contents.slideDown();
  }
  button.toggleClass("help_visible");
}

function toggleECSummaryHelpText() {
  let button = $("#learn_more_ec_button");
  let contents = $("#learn_more_ec_contents");

  toggleSummaryHelpText(button, contents);
}

function toggleGOSummaryHelpText() {
  let button = $("#learn_more_go_button");
  let contents = $("#learn_more_go_contents");

  toggleSummaryHelpText(button, contents);
}

/**
 * Switch to summary view, including both PDB and GO view.
 */
function switchToSummaryView() {
  makePDBAndGOSmall();

  bindCanvasHover();
  bindPDBHover();
  turnOnClickToEnterLargeModes();
  turnOffHoverActionsForCytoscape();
  hideCyPanzoom();

  hideLoaders();

  $('.magnifying_glass_text').show();
  $('.return_to_summary_mode_text').hide();

  // GO/cytoscape.
  $('#go_grid').show();
  $("#go_content_container").show();
  $('#go_container').show();
  $("#go_color_bar").hide();

  // EC/PDB.
  if (currentlyHasEnzymePrediction) {
    $('#enzyme_stuff').show();
    if (correspondingPDB != NO_PDB_STRUCTURE_SENTINEL) {
      $('#pdb_main_container').show();
    }

    $('#protvista').show();
    $("#ec_go_separator").show();

    $('#large_pdb_controls').hide();
    $('#large_ec_instructions').hide();
    $('#large_go_instructions').hide();
  } else {
    $("#enzyme_stuff").hide();
    $("#ec_go_separator").hide();
  }
  $("#ec_color_bar").hide();

  // Do this last because we want everything already resized before rerendering.
  // (but before disabling zooming altogether).
  rerenderPDBAndGO();

  disableInteractionsForSummaryView();
  disableNGLTooltip();
}

/**
 * Switch to PDB view (and hide GO contents).
 */
function switchToPDBView() {
  if (!currentlyHasEnzymePrediction) { // E.g. if switching from PDB view into hemoglobin.
    switchToSummaryView();
    return;
  }

  $('#protvista').hide();
  $('#viewonPDB').hide();
  $('#go_container').hide();
  $("#ec_loader").hide();

  $('#large_pdb_controls').show();
  $('#large_ec_instructions').show();
  $('.return_to_summary_mode_text').show();
  $("#ec_color_bar").show();
  $('#pdb_mag').hide();

  if (!pdbViewIsLarge()) {
    $('#pdb_view').removeClass('pdb_small_viewer');
    $('#pdb_view').addClass('pdb_large_viewer');

    pdbView.handleResize();

    enableInteractions();
    enableNGLTooltip();

    loadPDB(correspondingPDB);

    $('.magnifying_glass_text').hide();

    // Don't switchToPDBView if you're already here and you click - this interferes with other
    // interactions with ngl.
    $("#pdb_view").off('click');

    // Do this last because we want everything already resized before rerendering.
    rerenderPDBAndGO();
  }
}

/**
 * Switch to GO view (and hide EC contents).
 */
function switchToGOView() {
  
  if(goViewIsLarge()){
    // Detect if view is already large and do nothing.
    // (This shouldn't be necessary as the event handler for click should be cancelled but this seems to fail
    // for reasons unknown).
    console.log("View is already large.")
    return; 
  }
  $('#magnify_go').hide();
  console.log("Switching to GO view")
  $('#cy').removeClass('cy_small_viewer');
  $('#cy').addClass('cy_large_viewer');
  $("#ec_go_separator").hide();
  $("#go_loader").hide();

  $('#enzyme_stuff').hide();
  $("#view_go_container").hide();
  $("#go_grid").hide();

  $('#large_go_instructions').show();
  $('#go_content_container').show();
  $('#unswitch_to_go').show();
  $("#go_color_bar").show();


 
  enableInteractions();
  turnOnHoverActionsForCytoscape();
  showCyPanzoom();

  $('.magnifying_glass_text').hide();

  // Don't switchToGOView if you're already here and you click - this interferes with other
  // interactions with cytoscape.
  $("#cy").off('click');

  // Do this last because we want everything already resized before rerendering.
  rerenderPDBAndGO();
}

function bindCanvasHover(){
  $('#cy').mouseenter(function(){
    if(goViewIsLarge()){
      $('#magnify_go').hide();
    }
    else{
    $('#magnify_go').show();
    }
  });

  $('#cy').mouseleave(function(){
    $('#magnify_go').hide();
  })

}



function bindPDBHover(){
  $('#color_bar_and_pdb_view').mouseenter(function(){
    if(pdbViewIsLarge()){
      $('#pdb_mag').hide();
    }
    else{
    $('#pdb_mag').show();
    }
  });

  $('#color_bar_and_pdb_view').mouseleave(function(){
    $('#pdb_mag').hide();
  })

}
