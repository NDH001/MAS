from flask import Blueprint, render_template, jsonify, request
from app import app

api = Blueprint('api', __name__, template_folder='templates')

@api.route('/')
def api_index():
	return 'api homepage'

@api.route('/datatable_example')
def api_datatable_example():
	response = {}
	""" 
		1. deliberately left out location, 
		   students to make it work,
		2. make datatable display url as anchor in DT
	"""
		
	event_a = []
	event_a.append("WWCode Singapore's social coding event")
	event_a.append("Mon, 1 october, 6.30pm - 8:20pm")
	event_a.append("http://meetu.ps/e/FcVHX/8qCls/d")
	event_a.append("Robert Sim")
	
	event_b = []
	event_b.append("HackerspaceSG Plenum (tentative)")
	event_b.append("Wed, 10 October, 8:00pm – 8:30pm")
	event_b.append("https://hackerspace.sg/plenum/")
	event_b.append("HSG")
	
	response['data'] = []
	response['data'].append(event_a)
	response['data'].append(event_b)
	
	response = jsonify(response)

	return response
	
