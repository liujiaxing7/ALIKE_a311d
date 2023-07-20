#pragma once
#include "Tracker.h"
#include <iostream>

Tracker::Tracker()
{
}

Tracker::Tracker(int t_cnt, int b_cnt)
{
	top_count = t_cnt;
	bot_count = b_cnt;
}

Tracker::~Tracker()
{
}

void Tracker::get_match_result(long long** match_result) {

	for (size_t kc = 0; kc < keep_count; ++kc) {
		memcpy(match_result[kc], match_point[kc], sizeof(long long) *4);
	}


	//delete
	for (size_t i = 0; i < keep_count; i++) {
		delete[] match_point[i];
	}
	delete[] match_point;

	for (size_t i = 0; i < 3; i++) {
		delete[] matches[i];
	}
	delete[] matches;

	match_point = nullptr;
	matches = nullptr;
}

void Tracker::set_keep_count(long long val) {

	keep_count = val;

}

long long Tracker::get_keep_count() {

	return keep_count;

}

void Tracker::match_point_idx(long long** top_pt, long long** bot_pt){
	//match_twoway���� ���� matching�Ǵ� point�� �ε����� ������, ���� point��ǥ�� ��ȯ

	match_point = new long long* [keep_count];

	for (size_t kc = 0; kc < keep_count; kc++) {
		long long* point_4 = new long long[4];

		int top_idx = matches[0][kc];
		int bot_idx = matches[1][kc];

		point_4[0] = top_pt[0][top_idx]; //top x
		point_4[1] = top_pt[1][top_idx]; //top y
		point_4[2] = bot_pt[0][bot_idx]; //bot x
		point_4[3] = bot_pt[1][bot_idx]; //bot y

		match_point[kc] = point_4;
	}
		
}


void Tracker::match_twoway(double** top_desc, double** bot_desc) {

	//top_desc�� transpose�� bot_desc�� dot product
	double** dmat = new double* [top_count]; //dot product ��� ������ ��

	for (size_t tc = 0; tc < top_count; tc++) {
		double* dmat_ = new double[bot_count];
		for (size_t bc = 0; bc < bot_count; bc++) {
			double val = 0;
			for (size_t dc = 0; dc < desc_channel; dc++){
				val += top_desc[dc][tc] * bot_desc[dc][bc]; //dot product
			}
			//dot product ����� -1~1 ������ ������ clip
			if (val > 1) {
				val = 1;
			}
			else if (val < -1) {
				val = -1;
			}
			val = sqrt(2 - 2 * val);
			dmat_[bc] = val;
		}
		dmat[tc] = dmat_;
	}


	long long* min_idx = new long long[top_count]; // dmat�� �� �࿡���� �ּҰ� �ε���
	for (size_t tc = 0; tc < top_count; tc++) {
		double mval = dmat[tc][0];
		int midx = 0;
		for (size_t bc = 1; bc < bot_count; bc++) {
			if (mval > dmat[tc][bc]) {
				mval = dmat[tc][bc];
				midx = bc;
			}
		}
		min_idx[tc] = midx;
	}

	bool* keep01 = new bool[top_count];
	for (size_t tc = 0; tc < top_count; tc++) {
		//min_idx�� �������� ������ score = dmat[tc][min_idx[tc]]�� (�� �� �࿡���� �ּҰ���)
		//threshold���� ������ keep
		double score = dmat[tc][min_idx[tc]];
		if (score < nn_thresh) { 
			keep01[tc] = true;
		}
		else
			keep01[tc] = false;
	}

	long long* min_idx2 = new long long[bot_count]; //dmat�� �� �������� �ּҰ� �ε���
	for (size_t bc = 0; bc < bot_count; bc++) {
		double mval = dmat[0][bc];
		int midx = 0;
		for (size_t tc = 1; tc < top_count; tc++) {
			if (mval > dmat[tc][bc]) {
				mval = dmat[tc][bc];
				midx = tc;
			}
		}
		min_idx2[bc] = midx;
	}

	//��������� matching�� index ŵ.
	bool* keep_bi = new bool[top_count];
	for (size_t tc = 0; tc < top_count; tc++) {
		//min_idx2�� min_idx�� �����ϸ�, matching�� �´� �ֵ��� ���ڰ� �Ȱ��� 
		if (tc == min_idx2[min_idx[tc]])
			keep_bi[tc] = true;
		else
			keep_bi[tc] = false;
	}	


	//matching�� �ֵ鸸 keep02�� �ش� �ε����� true�� ����
	bool* keep02 = new bool[top_count];
	int keep_cnt = 0;
	for (size_t tc = 0; tc < top_count; tc++) {
		keep02[tc] = (keep01[tc] && keep_bi[tc]);
		if (keep02[tc] == true)
			keep_cnt++;
	}

	set_keep_count(keep_cnt); //matching�� ��ǥ ���� ����

	//matching�� �ֵ鸸 ����
	double* keep_min_idx = new double[keep_count]; //python->m_idx2
	double* keep_score = new double[keep_count]; //python -> scores
	int keepidx = 0;
	for (size_t tc = 0; tc < top_count; tc++) {
		if (keep02[tc] == true) {
			keep_min_idx[keepidx] = double(min_idx[tc]);
			keep_score[keepidx] = dmat[tc][min_idx[tc]];
			keepidx++;
		}
	}

	//matches -> matching�� ��ǥ�ε���, score ������ ��
	matches = new double* [3];
	for (size_t i = 0; i < 3; i++) {
		double* match = new double[keep_count];
		matches[i] = match;
	}

	double* match01 = new double[keep_count];
	int midx = 0;
	for (size_t tc = 0; tc < top_count; tc++) {
		if (keep02[tc] == true) {
			match01[midx] = double(tc);
			midx++;
		}
	}
	//��� �� ���� 
	memcpy(matches[0], match01, sizeof(double) * keep_count); //top point�� �ε���
	memcpy(matches[1], keep_min_idx, sizeof(double) * keep_count);// top point�� ��Ī�Ǵ� bot point �ε���
	memcpy(matches[2], keep_score, sizeof(double) * keep_count); //�� ��Ī�� score 

	//delete
	for (size_t i = 0; i < top_count; i++) {
		delete[] dmat[i];
	}
	delete[] dmat;
	delete[] min_idx;
	delete[] keep01;
	delete[] min_idx2;
	delete[] keep_bi;
	delete[] keep02;
	delete[] keep_min_idx;
	delete[] keep_score;

	dmat = nullptr;
	min_idx = nullptr;
	keep01 = nullptr;
	min_idx2 = nullptr;
	keep_bi = nullptr;
	keep02 = nullptr;
	keep_min_idx = nullptr;
	keep_score = nullptr;
}