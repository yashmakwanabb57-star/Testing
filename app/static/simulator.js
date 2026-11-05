// Configurator simulator: a port of the shop's coloring logic in
// reference/shop-configurator.js (globalMethods.setSvgImgColor and
// setSvgImgRoseVisibility). The class loop, selectors, IDs, pattern sizing and layer
// toggles are kept as in the shop code. Adaptations are marked "ADAPTED".

const SHOP_CLASSES = ['st0', 'st1', 'st2', 'st3', 'st5'];
export const COLOR_INDICES = [1, 2, 3, 4];

// Port of setSvgImgColor (shop-configurator.js lines 2-62).
// ADAPTED: takes the svg element, the color index and a color object
// {value, productId, imageWidth, imageHeight} as arguments instead of a Vue node and
// a lookup in window.eee_productConfigColors; the two window.eee_* globals live in
// the `state` object so one applyConfig call is one simulated page session.
function setSvgImgColor(svgImg, colorIndex, color, state) {
  colorIndex = Number(colorIndex);
  const value = color ? color.value : '';
  const productId = color ? color.productId : '';
  const imageWidth = color ? color.imageWidth : 0;
  const imageHeight = color ? color.imageHeight : 0;

  if (!svgImg || !colorIndex) {
    return;
  }

  const image = value.startsWith('url(') ? value.slice(4, -1).replace(/["']/g, '') : '';
  if (productId || image) {
    state.showPattern = true;
    state.patternColorIndex = colorIndex;
  } else if (state.patternColorIndex === colorIndex) {
    state.showPattern = false;
  }
  for (const className of SHOP_CLASSES) {
    const patternSuffix = productId || image ? '_pattern' : '';
    const elements = svgImg.querySelectorAll(`.${className}_${colorIndex}${patternSuffix}`);
    if (elements.length > 0) {
      if (image) {
        const patternElement = svgImg.querySelector(`#image_${className}_${colorIndex}_pattern`);
        const imageElement = patternElement ? patternElement.querySelector(`image`) : null;
        if (imageElement) {
          patternElement.setAttribute('width', imageWidth || 80);
          patternElement.setAttribute('height', imageHeight || 80);
          imageElement.setAttribute('width', imageWidth || 80);
          imageElement.setAttribute('height', imageHeight || 80);
          imageElement.setAttribute('xlink:href', image);
          for (const element of elements) {
            element.style.fill = `url(#image_${className}_${colorIndex}_pattern)`;
          }
        }
      } else {
        for (const element of elements) {
          element.style.fill = value;
        }
      }
    }
  }

  const patternLayer = svgImg.querySelector('#Layer_3');
  const solidLayer = svgImg.querySelector('#Layer_2_1');
  if (patternLayer && solidLayer) {
    patternLayer.style.display = state.showPattern ? 'inline' : 'none';
    solidLayer.style.display = state.showPattern ? 'none' : 'inline';
  }

  // Kept from the shop code (its own comment: "no idea why it was needed").
  const layer2 = svgImg.querySelector('#Layer_2');
  if (layer2) {
    layer2.style.stroke = 'none';
  }
}

// Port of setSvgImgRoseVisibility (shop-configurator.js lines 64-81).
// ADAPTED: takes showRose directly instead of reading the 'Rose' option row of a node.
function setSvgImgRoseVisibility(svgImg, showRose) {
  if (!svgImg) {
    return;
  }
  const roseElements = svgImg.querySelectorAll('.st4');
  const circleElements = svgImg.querySelectorAll('.st_circel');
  for (const element of roseElements) {
    element.style.display = showRose ? 'inline' : 'none';
  }
  for (const element of circleElements) {
    element.style.display = showRose ? 'none' : 'inline';
  }
}

// Apply a full configurator state to an inline <svg> element.
// config: {colors: {1:'#hex', 2:..., 3:..., 4:...}, patternIndex: 2|null,
//          patternImage: dataUrl|null, rose: bool}
// ADAPTED: the shop calls setSvgImgColor once per changed option; here every color
// index is applied in order 1..4 on a fresh SVG, as if each option had been picked
// once. A patterned color is passed the way the shop gets it from its color list: a
// value of the form url(<image>), no productId, no image size (so the size is 80).
export function applyConfig(svgEl, config) {
  const state = { showPattern: false, patternColorIndex: undefined };
  const colors = config.colors || {};
  for (const i of COLOR_INDICES) {
    const usePattern = config.patternIndex === i && config.patternImage;
    const value = usePattern ? `url(${config.patternImage})` : (colors[i] || '');
    setSvgImgColor(svgEl, i, { value, productId: '', imageWidth: 0, imageHeight: 0 }, state);
  }
  setSvgImgRoseVisibility(svgEl, !!config.rose);
}

// Built-in sample pattern: an 80x80 glitter tile drawn on a canvas from a base color.
// Deterministic (seeded) so the preview does not flicker between redraws.
export function glitterPattern(baseHex, size = 80) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = baseHex || '#9fb7c4';
  ctx.fillRect(0, 0, size, size);
  let seed = 12345;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let k = 0; k < 220; k++) {
    const x = rand() * size;
    const y = rand() * size;
    const r = 0.4 + rand() * 1.3;
    const light = rand() < 0.65;
    const a = 0.35 + rand() * 0.6;
    ctx.fillStyle = light ? `rgba(255,255,255,${a.toFixed(2)})` : `rgba(0,0,0,${(a * 0.5).toFixed(2)})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  return canvas.toDataURL('image/png');
}
