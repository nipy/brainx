==========================================
 Pre-processing steps in Rest.TMS dataset
==========================================

The data lives in::

  /r/d2/despo/enomura/data/Rest.Lesion/Data/NNN

where NNN runs 101..122.  Each directory has a subdir like 20090402/ and one
called Total/ that contains the processed data in a NIfTI/ subdir.

The TMS data is similarly organized, but in::

  /home/despo/cgratton/data/Rest.TMS/Data

The following code (in red) was used to preprocess the Rest.TMS data.  Most of
this was executed with Alloy, which is a MATLAB script that sends out AFNI
commands to the shell.  The only SPM step is normalizing.  AFNI has an
auto-normalize command, but it writes out files in talairach space, making it
difficult to use the AAL atlas.

1.  DICOM conversion -- Anatomical scans::

    to3d -anat -prefix 101-T1.nii *.dcm
    3drefit -deoblique -xorigin cen -yorigin cen -zorigin cen 101-T1.nii
    3dresample -orient RPI -prefix 101-T1.nii -inset 101-T1.nii

2.  DICOM conversion -- Functional scans::

    to3d -epan -skip_outliers -assume_dicom_mosaic -time:zt 24 435 2 alt+z2 -prefix 101-EPI-001.nii *.dcm
    3drefit -deoblique -xorigin cen -yorigin cen -zorigin cen 101-EPI-001.nii
    3dresample -orient RPI -prefix 101-EPI-001.nii -inset 101-EPI-001.nii

3.  Generate Mean image::

    3dTstat -prefix 101-Mean.nii 101-EPI-001.nii

4.  Generate Mask image::

    3dAutomask -prefix 101-Mask.nii 101-Mean.nii

5.  Volume registration::
    
    3dvolreg -twodup -verbose -tshift 0 -base 101-Mean.nii -maxdisp1D 101-MD1D.txt -1Dfile 101-EPI-001-1D.txt -prefix 101-EPI-001-CoReg.nii 101-EPI-001.nii

[Note: all blocks are registered to the Mean image, which is the first block.
This block is also used to coregister the anatomical image]

6.  Smoothing::

    3dmerge -doall -1blur_fwhm 5 -prefix 101-EPI-001-CoReg-Smooth.nii 101-EPI-001-CoReg.nii

7.  Anatomical alignment::

    3dZcutup -keep 80 240 -prefix 101-T1-Cut.nii 101-T1.nii
    3drefit -deoblique -xorigin cen -yorigin cen -zorigin cen 101-T1-Cut.nii
    3dresample -orient RPI -prefix 101-T1-Cut.nii -inset 101-T1-Cut.nii

    lpc_align.py -epi 101-Mean.nii -anat 101-T1-Cut.nii -strip_anat_skull no -suffix -CoReg

[Note: the zcutup command trims the image so that it only contains the brain.
The following commands then recenter the brain and make sure it is oriented
properly]

8.  Normalize

Create a centered SPM T1.nii template image that is centered (just done once)::

    3drefit -xorigin cen -yorigin cen -zorigin cen T1.nii
   
MATLAB/SPM method:
        Click on 'Normalize'
        Use the *sn.mat file generated previously to write out the T1 file
        Uses the 'T1_center' template, which puts the brain in MNI space
        Write out a voxel size of 1x1x1           
        
AFNI method:
        @auto_tlrc
        [fill more in later...for now we are not using this method]

9.  Atlas templates
    
    The AAL template:
        original file is not centered, is oriented LPI and is an analyze file

Commands::

        3dcopy aal.hdr aal.nii
        3drefit -xorigin cen -yorigin cen -zorigin cen aal.nii
        3drefit -orient RPI aal.nii

Resample aal to match the T1 centered template (just done once) (it should be
in cgratton/data/Rest.TMS/Data/Masks/norm_mni_aal).  Then::
       
        3dresample -master T1_center.nii -prefix aal_r.nii -inset aal.nii

10.  reverse normalize using the sn.mat file created in step 8 (this is the
script Emi wrote, which we can use

For the AAL template, use the aal_r.nii ROI and the appropriate version of the
script (reversenorm_tmsrest_spm5_aal)

For the Dosenbach ROIs, change the directory to 'norm_mni_spm5' (check this)

        Steps in the reverse normalize script:
       
        i.  load the EPI file in native space
        ii.  find where it is non-zero
        iii.  find the size of the normal space ROI
        iv.  get the coordinates of the ROI in mm in native space
        v.  find the coordinates of these in voxel coordinates in native space
        vi. 

For the lesion patients:
8a. Segment to create the sn.mat file

Move the coordinates within the sn.mat file to match the centered template
