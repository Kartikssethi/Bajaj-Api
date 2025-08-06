const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

// Test the new file upload endpoint
async function testFileUpload() {
  const baseURL = 'http://localhost:3000';
  
  // Test questions
  const questions = [
    "What is the main topic of this document?",
    "What are the key points mentioned?",
    "Are there any important dates or numbers mentioned?"
  ];

  console.log('🧪 Testing File Upload Endpoint');
  console.log('================================');

  // Test 1: PDF file (if you have one)
  console.log('\n📄 Test 1: PDF Processing');
  console.log('Note: Create a test.pdf file in the same directory to test this');
  
  if (fs.existsSync('./test.pdf')) {
    try {
      const formData = new FormData();
      formData.append('file', fs.createReadStream('./test.pdf'));
      formData.append('questions', JSON.stringify(questions));

      const response = await axios.post(`${baseURL}/hackrx/upload`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 60000, // 60 seconds timeout
      });

      console.log('✅ PDF processed successfully');
      console.log('📊 Metadata:', response.data.metadata);
      console.log('💬 Answers:', response.data.answers);
    } catch (error) {
      console.error('❌ PDF test failed:', error.response?.data || error.message);
    }
  } else {
    console.log('⚠️  No test.pdf file found - skipping PDF test');
  }

  // Test 2: Word document (if you have one)
  console.log('\n📝 Test 2: Word Document Processing');
  console.log('Note: Create a test.docx file in the same directory to test this');
  
  if (fs.existsSync('./test.docx')) {
    try {
      const formData = new FormData();
      formData.append('file', fs.createReadStream('./test.docx'));
      formData.append('questions', JSON.stringify(questions));

      const response = await axios.post(`${baseURL}/hackrx/upload`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 60000,
      });

      console.log('✅ Word document processed successfully');
      console.log('📊 Metadata:', response.data.metadata);
      console.log('💬 Answers:', response.data.answers);
    } catch (error) {
      console.error('❌ Word document test failed:', error.response?.data || error.message);
    }
  } else {
    console.log('⚠️  No test.docx file found - skipping Word document test');
  }

  // Test 3: Excel file (if you have one)
  console.log('\n📊 Test 3: Excel Spreadsheet Processing');
  console.log('Note: Create a test.xlsx file in the same directory to test this');
  
  if (fs.existsSync('./test.xlsx')) {
    try {
      const formData = new FormData();
      formData.append('file', fs.createReadStream('./test.xlsx'));
      formData.append('questions', JSON.stringify([
        "What data is in the first sheet?",
        "What are the column headers?",
        "What is the total number of rows?"
      ]));

      const response = await axios.post(`${baseURL}/hackrx/upload`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 60000,
      });

      console.log('✅ Excel file processed successfully');
      console.log('📊 Metadata:', response.data.metadata);
      console.log('💬 Answers:', response.data.answers);
    } catch (error) {
      console.error('❌ Excel test failed:', error.response?.data || error.message);
    }
  } else {
    console.log('⚠️  No test.xlsx file found - skipping Excel test');
  }

  // Test 4: Image with OCR (if you have one)
  console.log('\n🖼️  Test 4: Image OCR Processing');
  console.log('Note: Create a test-image.jpg file in the same directory to test this');
  
  if (fs.existsSync('./test-image.jpg')) {
    try {
      const formData = new FormData();
      formData.append('file', fs.createReadStream('./test-image.jpg'));
      formData.append('questions', JSON.stringify([
        "What text is visible in this image?",
        "Are there any numbers or dates visible?",
        "What is the main content of this image?"
      ]));

      const response = await axios.post(`${baseURL}/hackrx/upload`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 120000, // 2 minutes for OCR
      });

      console.log('✅ Image OCR processed successfully');
      console.log('📊 Metadata:', response.data.metadata);
      console.log('💬 Answers:', response.data.answers);
    } catch (error) {
      console.error('❌ Image OCR test failed:', error.response?.data || error.message);
    }
  } else {
    console.log('⚠️  No test-image.jpg file found - skipping image OCR test');
  }

  console.log('\n🎯 Test completed!');
  console.log('\n📋 Usage Instructions:');
  console.log('1. Start the server: npm start');
  console.log('2. Use the /hackrx/upload endpoint with multipart/form-data');
  console.log('3. Include a "file" field with your document');
  console.log('4. Include a "questions" field with JSON array of questions');
  console.log('5. Supported formats: PDF, Word (.doc/.docx), Excel (.xls/.xlsx), Images');
}

// Run the test
testFileUpload().catch(console.error); 