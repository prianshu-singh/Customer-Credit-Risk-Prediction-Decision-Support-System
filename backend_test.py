import requests
import sys
import json
import io
import pandas as pd
from datetime import datetime

class CreditRiskAPITester:
    def __init__(self, base_url="https://debtwatch.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.token = None
        self.tests_run = 0
        self.tests_passed = 0
        self.customer_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    # Remove Content-Type for file uploads
                    headers.pop('Content-Type', None)
                    response = requests.post(url, files=files, headers=headers)
                else:
                    response = requests.post(url, json=data, headers=headers)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test health endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "health",
            200
        )
        return success

    def test_register_user(self):
        """Test user registration"""
        test_user_data = {
            "username": "testuser",
            "email": "test@example.com", 
            "password": "testpass123"
        }
        
        success, response = self.run_test(
            "User Registration",
            "POST",
            "register",
            200,
            data=test_user_data
        )
        return success

    def test_login_user(self):
        """Test user login and get token"""
        login_data = {
            "username": "testuser",
            "password": "testpass123"
        }
        
        success, response = self.run_test(
            "User Login",
            "POST",
            "login",
            200,
            data=login_data
        )
        
        if success and 'access_token' in response:
            self.token = response['access_token']
            print(f"   Token obtained: {self.token[:20]}...")
            return True
        return False

    def test_upload_customer_data(self):
        """Test CSV file upload"""
        # Create sample customer data with at least 5 customers for ML training
        sample_data = {
            'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005', 'CUST006'],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson', 'Diana Lee'],
            'age': [35, 28, 45, 32, 55, 29],
            'annual_income': [50000, 75000, 40000, 60000, 80000, 45000],
            'credit_score': [650, 720, 580, 680, 750, 620],
            'debt_to_income_ratio': [0.3, 0.2, 0.5, 0.25, 0.15, 0.4],
            'employment_length': [5, 3, 10, 7, 15, 2],
            'loan_amount': [25000, 30000, 15000, 35000, 40000, 20000],
            'loan_purpose': ['home', 'auto', 'personal', 'home', 'auto', 'personal'],
            'home_ownership': ['rent', 'own', 'mortgage', 'own', 'own', 'rent']
        }
        
        df = pd.DataFrame(sample_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        files = {
            'file': ('test_customers.csv', csv_content, 'text/csv')
        }
        
        success, response = self.run_test(
            "Upload Customer Data",
            "POST",
            "upload-customer-data",
            200,
            files=files
        )
        
        if success:
            self.customer_id = 'CUST001'  # Store for later tests
        
        return success

    def test_get_customers(self):
        """Test getting customer list"""
        success, response = self.run_test(
            "Get Customers",
            "GET",
            "customers",
            200
        )
        return success

    def test_analyze_customer_risk(self):
        """Test risk analysis for a customer"""
        if not self.customer_id:
            print("❌ No customer ID available for risk analysis")
            return False
            
        success, response = self.run_test(
            "Analyze Customer Risk",
            "POST",
            f"analyze-risk/{self.customer_id}",
            200
        )
        
        if success:
            print(f"   Risk Score: {response.get('risk_score', 'N/A')}")
            print(f"   Risk Level: {response.get('risk_level', 'N/A')}")
            print(f"   AI Analysis: {response.get('ai_analysis', 'N/A')[:100]}...")
        
        return success

    def test_get_risk_predictions(self):
        """Test getting risk predictions"""
        success, response = self.run_test(
            "Get Risk Predictions",
            "GET",
            "risk-predictions",
            200
        )
        return success

    def test_get_alerts(self):
        """Test getting alerts"""
        success, response = self.run_test(
            "Get Alerts",
            "GET",
            "alerts",
            200
        )
        return success

    def test_dashboard_stats(self):
        """Test dashboard statistics"""
        success, response = self.run_test(
            "Dashboard Statistics",
            "GET",
            "dashboard-stats",
            200
        )
        
        if success:
            print(f"   Total Customers: {response.get('total_customers', 0)}")
            print(f"   Total Predictions: {response.get('total_predictions', 0)}")
            print(f"   Active Alerts: {response.get('active_alerts', 0)}")
        
        return success

def main():
    print("🚀 Starting Credit Risk API Testing...")
    print("=" * 60)
    
    tester = CreditRiskAPITester()
    
    # Test sequence
    tests = [
        ("Health Check", tester.test_health_check),
        ("User Registration", tester.test_register_user),
        ("User Login", tester.test_login_user),
        ("Upload Customer Data", tester.test_upload_customer_data),
        ("Get Customers", tester.test_get_customers),
        ("Dashboard Statistics", tester.test_dashboard_stats),
        ("Analyze Customer Risk", tester.test_analyze_customer_risk),
        ("Get Risk Predictions", tester.test_get_risk_predictions),
        ("Get Alerts", tester.test_get_alerts),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            if not success:
                failed_tests.append(test_name)
        except Exception as e:
            print(f"❌ {test_name} - Exception: {str(e)}")
            failed_tests.append(test_name)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Tests Run: {tester.tests_run}")
    print(f"Tests Passed: {tester.tests_passed}")
    print(f"Tests Failed: {len(failed_tests)}")
    print(f"Success Rate: {(tester.tests_passed/tester.tests_run*100):.1f}%")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"   - {test}")
    else:
        print(f"\n✅ All tests passed!")
    
    return 0 if len(failed_tests) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())