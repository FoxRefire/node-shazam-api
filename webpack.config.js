const path = require('path');  
  
module.exports = {  
  entry: './src/index.ts',  
  mode: 'production',  
  module: {  
    rules: [  
      {  
        test: /\.tsx?$/,  
        use: 'ts-loader',  
        exclude: [/node_modules/, /dist/],
        include: [/src/]
      },  
    ],  
  },  
  resolve: {  
    extensions: ['.tsx', '.ts', '.js'],  
    fallback: {  
      "http": false,  
      "https": false,  
      "url": false,  
      "buffer": false,  
      "stream": false  
    }  
  },  
  output: {  
    filename: 'shazam-api.min.js',  
    path: path.resolve(__dirname, 'webpack-dist'),  
    library: {
      name: 'ShazamAPI',
      type: 'self',  
    }, 
  },  
  optimization: {
    minimize: true,
  },  
  experiments: {
    outputModule: true,
  },
  target: 'web'
};