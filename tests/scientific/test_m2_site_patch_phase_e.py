"""Phase E opt-in site/patch scientific guardrails. Synthetic and model-free."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lithiumscope.model_2.methodology.contracts import sample_contract
from lithiumscope.model_2.methodology.site_patch_lab import (
    ARMS,
    aggregate_training_sites,
    finite_quality_filter,
    half_patch_predictors,
    historical_train_groups,
    irreducible_identical_input_minority,
    describe_sample_subsets,
    site_level_evaluation,
)


@pytest.fixture
def contract():
    return sample_contract(pd.DataFrame({
        "sample_id":list("ABCDEFGH"),
        "image_path":[f"p{i}.npy" for i in range(8)],
        "Li_icpms":[10.,40.,20.,50.,15.,30.,22.,35.],
        "latitude":[-30.,-30.,-31.,-31.,-32.,-33.,-34.,-35.],
        "longitude":[-70.,-70.,-70.,-70.,-70.,-70.,-70.,-70.],
        "spatial_group":["g0","g0","g1","g1","g2","g3","g4","g5"],
        "sentinel_scene_id":[f"scene{i}" for i in range(8)],
    }))


@pytest.fixture
def spectra():
    return pd.DataFrame({
        "red_mean":[10.,10.,11.,11.,12.,13.,14.,15.],
        "nir_mean":[1.,1.,2.,2.,3.,4.,5.,6.],
    })


@pytest.fixture
def image_hashes():
    return pd.Series(["same0","same0","same1","same1","p4","p5","p6","p7"])


def test_pre_registered_arms_not_downstream_modified():
    assert len(ARMS)==6
    assert ARMS[0]=="historical_706_reference"
    assert ARMS[-1]=="central_16x16_patch"
    assert "site_majority_consensus" in ARMS


def test_site_median_is_one_site_per_train_and_no_test_li_access(
    contract,spectra,image_hashes
):
    before=contract.copy(deep=True)
    selected=np.array([0,1,2,3,4,5])
    x,y,info=aggregate_training_sites(
        contract,spectra,selected,lithium_threshold=25.,
        policy="median",sample_sha256=image_hashes
    )
    assert len(x)==4 and len(y)==4 and len(info)==4
    # Median sites: 10/40 =25 high, 20/50=35 high, 15 low, 30 high.
    assert y.tolist()==[1,1,0,1]
    assert_frame_equal(before,contract)
    # Change held-out Li: neither site labels nor training input may change.
    mutated=contract.copy(deep=True)
    mutated.loc[6:7,"Li_icpms"]=[1000.,0.]
    x2,y2,info2=aggregate_training_sites(
        mutated,spectra,selected,lithium_threshold=25.,
        policy="median",sample_sha256=image_hashes
    )
    assert_frame_equal(x,x2)
    assert np.array_equal(y,y2)
    assert_frame_equal(info,info2)


def test_majority_with_tie_abstains_in_train_only(contract,spectra,image_hashes):
    X,y,info=aggregate_training_sites(
        contract,spectra,np.arange(6),lithium_threshold=25.,
        policy="majority",sample_sha256=image_hashes
    )
    assert len(X)==2
    assert y.tolist()==[0,1]
    assert info.included.tolist()==[False,False,True,True]
    assert (info.loc[~info.included,"decision"]=="EXCLUDE_TIED_SITE_TRAIN_ONLY").all()


def test_site_patch_hash_collision_is_not_silently_aggregated(
    contract,spectra,image_hashes
):
    mismatched=image_hashes.copy()
    mismatched.iloc[1]="changed_image"
    with pytest.raises(ValueError,match="different spectral patches"):
        aggregate_training_sites(
            contract,spectra,np.arange(6),lithium_threshold=25.,
            policy="median",sample_sha256=mismatched
        )


def test_predefined_quality_filter_never_excludes_test_samples():
    q=pd.DataFrame({"finite_pixel_fraction":[1.,.52,.99,.93,1.,.7]})
    original=np.array([0,1,2,3])
    test=np.array([4,5])
    filtered=finite_quality_filter(original,q)
    assert filtered.tolist()==[0,2]
    assert test.tolist()==[4,5]
    assert original.tolist()==[0,1,2,3]
    assert finite_quality_filter(original,q,minimum_fraction=.5).tolist()==[0,1,2,3]
    with pytest.raises(ValueError,match="quality threshold"):
        finite_quality_filter(original,q,minimum_fraction=0)


def test_irreducible_mixed_site_errors_use_fold_train_q75(
    contract,image_hashes,monkeypatch
):
    from lithiumscope.model_2.methodology import site_patch_lab as lab
    monkeypatch.setattr(lab,"define_outer_fold",
        lambda fold,manifest:(np.array([0,1,2,3,4,5]),
          np.array([0,1,2,3]),np.array([4,5,6,7]),25.))
    fold={"train_indices":[0,1,2,3,4,5],"test_indices":[6,7]}
    qc=pd.DataFrame({"image_sha256":image_hashes})
    bound=irreducible_identical_input_minority(contract,[fold],qc)
    assert bound.iloc[0].partition=="train"
    assert bound.iloc[0].mixed_sites==2
    assert bound.iloc[0].minimum_unavoidable_binary_errors==2
    assert bound.iloc[1].minimum_unavoidable_binary_errors==0


def test_center_patch_same_sha_avoids_rereading_and_fixed_schema(
    tmp_path,contract
):
    from lithiumscope.model_2.steps.step_05_spectral_features import (
        extract_spectral_features
    )
    from lithiumscope.model_2.steps.step_04_image_preprocessing import preprocess_image
    arr=np.zeros((6,32,32),dtype=np.float32)
    arr[0,:,:]=1.
    arr[0,8:24,8:24]=2.
    arr[3,:,:]=5.
    image=tmp_path/"sample.npy"
    np.save(image,arr,allow_pickle=False)
    digest=hashlib.sha256(image.read_bytes()).hexdigest()
    c=contract.iloc[:2].copy()
    c["image_path"]=str(image)
    qc=pd.DataFrame({"image_sha256":[digest,digest]})
    full=extract_spectral_features(preprocess_image(arr))
    output=half_patch_predictors(c,list(full),qc)
    assert output.shape==(2,126)
    assert output.iloc[0].equals(output.iloc[1])
    assert output.loc[0,"red_mean"]==pytest.approx(2.,abs=1e-5)
    assert output.loc[0,"nir_mean"]==pytest.approx(5.,abs=1e-5)


def test_site_level_metrics_fixed_labels_across_arms():
    rows=[]
    for arm,scores in [
        ("historical_706_reference",[.1,.1,.8,.8,.4,.7]),
        ("site_median_li",[.2,.2,.7,.7,.5,.6]),
    ]:
        for i in range(6):
            rows.append({
                "arm":arm,"fold":1 if i<2 else 2,
                "site_screening_id":["site1","site1","site2","site2","site3","site4"][i],
                "sample_id":str(i),"li_ppm":[10.,40.,20.,50.,15.,30.][i],
                "fold_li_threshold_ppm":25.,
                "y_true":[0,1,0,1,0,1][i],
                "score":scores[i],
            })
    result=site_level_evaluation(pd.DataFrame(rows))
    allsites=result[result.subset=="all_sites"]
    assert len(allsites)==2
    assert allsites.n_sites.eq(4).all()
    assert result[result.subset=="ambiguous_sites"].n_sites.eq(2).all()


def test_quality_subgroups_have_same_test_rows_per_arm():
    rows=[]
    for arm in ["historical_706_reference","site_median_li"]:
        for i in range(6):
            rows.append({"arm":arm,"test_index":i,
                         "site_screening_id":["a","a","b","b","c","d"][i],
                         "y_true":[0,1,0,1,0,1][i],
                         "score":(.1+.15*i),
                         "sample_id":str(i)})
    qc=pd.DataFrame({"finite_pixel_fraction":[.5,1,1,1,.8,1]})
    result=describe_sample_subsets(pd.DataFrame(rows),qc)
    assert result.loc[result.subset=="test_quality_below_95pct","n"].tolist()==[2,2]
    assert result.loc[result.subset=="all_706","n"].tolist()==[6,6]



from lithiumscope.model_2.methodology.patch_missingness import inspect_mask


def test_missing_mask_anatomy_distinguishes_shared_footprint_from_band_issue():
    whole=np.ones((6,32,32),dtype=np.float32)
    whole[:,:16,:]=np.nan
    observed=inspect_mask(whole)
    assert observed["all_six_bands_have_same_pixel_mask"] is True
    assert observed["finite_fraction"]==0.5
    assert observed["finite_center_fraction"]==0.5
    assert observed["finite_border_fraction"]==0.5
    assert observed["nonfinite_xmin"]==0
    assert observed["nonfinite_ymin"]==0
    isolated=whole.copy()
    isolated[0,17,17]=np.nan
    altered=inspect_mask(isolated)
    assert altered["all_six_bands_have_same_pixel_mask"] is False
    assert altered["fraction_pixels_sharing_validity_mask"]<1


def test_missing_mask_central_and_border_diverge():
    image=np.ones((6,32,32),dtype=np.float32)
    image[:,:8,:]=np.nan
    m=inspect_mask(image)
    assert m["finite_center_fraction"]==1.
    assert m["finite_border_fraction"]<1.
    with pytest.raises(ValueError,match="32x32"):
        inspect_mask(np.ones((6,16,16),dtype=float))
