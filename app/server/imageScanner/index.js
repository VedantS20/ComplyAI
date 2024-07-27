import ImageKit from 'imagekit';
import axios from 'axios';
import imageScannerUrls from '../collections';

const imagekit = new ImageKit({
  publicKey: 'public_UghEp/T158eyYtp7mtkhVRQkrew=',
  privateKey: 'private_216cw3c0/5sUx2KW5VJ3ELqRfgM=',
  urlEndpoint: 'https://ik.imagekit.io/4rp923pit',
});

Meteor.methods({
  uploadImage(file) {
    return imagekit.upload({
      file,
      fileName: `${+new Date()}-scannedimage.jpg`,
      extensions: [
        {
          name: 'google-auto-tagging',
          maxTags: 5,
          minConfidence: 95,
        },
      ],
    });
  },
  getAIResponseForScannedImage(imageURL) {
    const urlObject = imageScannerUrls.findOne({});
    return new Promise(function (resolve, reject) {
      axios(
        `${Meteor.settings.public.API_HOST}/imageRecognition?imageURL=${imageURL}&temperature=${urlObject.temperature}`,
        {
          method: 'POST',
        },
      )
        .then(result => {
          if (result?.data) {
            resolve(result.data);
          } else {
            reject();
          }
        })
        .catch(error => {
          reject(error);
        });
    });
  },
  getLatestUrlForImageScanning() {
    const urlObject = imageScannerUrls.findOne({});
    return urlObject.url;
  },
  saveUrl(url) {
    imageScannerUrls.remove({});
    imageScannerUrls.insert({
      url,
      createdOn: new Date(),
    });
    return imageScannerUrls.findOne({}).url;
  },
  saveTemp(temp) {
    imageScannerUrls.remove({});
    imageScannerUrls.insert({
      temperature: parseFloat(temp),
      createdOn: new Date(),
    });
    return imageScannerUrls.findOne({}).temperature;
  },
  getTemp() {
    return imageScannerUrls.findOne({}).temperature;
  },
});
