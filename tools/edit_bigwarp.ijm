match_dir = getDirectory("choose csv directory");
thumb_dir = getDirectory("choose thumbnail directory");
delimiter =  "(__to__)";
mlist = getFileList(match_dir);
for (i=0; i<mlist.length; i++) {
	name_ext = split(mlist[i], '.');
	section_names = split(name_ext[0], delimiter);
	if (lengthOf(section_names) < 2){
		continue;
	};
	section_file0 = section_names[0] + ".png";
	section_file1 = section_names[1] + ".png";
	open(thumb_dir + "/" + section_file0);
	selectWindow(section_file0);
	run("Red");
	open(thumb_dir + "/" + section_file1);
	selectWindow(section_file1);
	run("Cyan");
	if (name_ext[1] == "csv"){
		landmarks = match_dir + "/" + mlist[i];
		run("Big Warp", "browse=" + landmarks
					  + " moving_image=" + section_file0
					  + " target_image=" + section_file1
					  + " moving=[] moving_0=[] target=[] target_0=[]"
					  + " landmarks=" + landmarks);
	} else {
		run("Big Warp", "moving_image=" + section_file0
					  + " target_image=" + section_file1
					  + " moving=[] moving_0=[] target=[] target_0=[]"
					  + " landmarks=[]");
	}
	// Interact with Big Warp interface, and the user saves (replaces) the csv file
	// Then close Big Warp, and click "ok" to go to the next match
	waitForUser("current match", mlist[i]);
	selectWindow(section_file0);
	close();
	selectWindow(section_file1);
	close();
}
