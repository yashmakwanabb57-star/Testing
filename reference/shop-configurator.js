vueApp.config.globalProperties.globalMethods = {
  setSvgImgColor: (node, colorIndex) => {
    colorIndex = Number(colorIndex);
    const svgImg = document.querySelector('#product-config-svg svg');
    const color = vueApp.config.globalProperties.globalMethods.getColorByNode(node);
    const value = color ? color.value : '';
    const productId = color ? color.productId : '';
    const imageWidth = color ? color.imageWidth : 0;
    const imageHeight = color ? color.imageHeight : 0;

    // set "Additional Price Product" column value
    node['Additional Price Product'] = color && color.productId ? color.productId : '';

    if (!svgImg || !colorIndex) {
      return;
    }

    const image = value.startsWith('url(') ? value.slice(4, -1).replace(/["']/g, '') : '';
    if (productId || image) {
      window.eee_productConfigSvgShowPattern = true;
      window.eee_productConfigSvgPatternColorIndex = colorIndex;
    } else if (window.eee_productConfigSvgPatternColorIndex === colorIndex) {
      window.eee_productConfigSvgShowPattern = false;
    }
    for (const className of ['st0', 'st1', 'st2', 'st3', 'st5']) {
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
      patternLayer.style.display = window.eee_productConfigSvgShowPattern ? 'inline' : 'none';
      solidLayer.style.display = window.eee_productConfigSvgShowPattern ? 'none' : 'inline';
    }

    // They had this in their old code, no idea why it was needed (and if it's still needed)
    const layer2 = svgImg.querySelector('#Layer_2');
    if (layer2) {
      layer2.style.stroke = 'none';
    }
  },

  setSvgImgRoseVisibility: (node) => {
    const svgImg = document.querySelector('#product-config-svg svg');

    if (!svgImg) {
      return;
    }

    const roseOption = node.csvRows.find(child => child.Type == 'Option' && child.Value == 'Rose');
    const showRose = roseOption && roseOption.Selected === 'True';
    const roseElements = svgImg.querySelectorAll('.st4');
    const circleElements = svgImg.querySelectorAll('.st_circel');
    for (const element of roseElements) {
      element.style.display = showRose ? 'inline' : 'none';
    }
    for (const element of circleElements) {
      element.style.display = showRose ? 'none' : 'inline';
    }
  },

  getColorByNode: (node) => {
    const optionsNode = node.csvRows.find(child => child.Type == 'OptionsColor');
    const fileName = optionsNode && optionsNode.Rule || '';
    const color = fileName && window.eee_productConfigColors && window.eee_productConfigColors[fileName] ? window.eee_productConfigColors[fileName].find(color => color.label === node.Value) : null;
    return color;
  }
};
